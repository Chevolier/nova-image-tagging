import boto3
import json
import base64
import os
import pandas as pd
import requests
import time
from io import BytesIO
from PIL import Image

# AWS Credentials Configuration
# Option 1: Set your AWS credentials directly here (not recommended for production)
AWS_ACCESS_KEY_ID = None  # Replace with your access key or set to None to use default credentials
AWS_SECRET_ACCESS_KEY = None  # Replace with your secret key or set to None to use default credentials

# Option 2: Use environment variables (recommended)
# Set these environment variables in your system:
# export AWS_ACCESS_KEY_ID=your_access_key
# export AWS_SECRET_ACCESS_KEY=your_secret_key

# Option 3: Use AWS credentials file (~/.aws/credentials) - default behavior if above are None

def detect_image_format(image_path):
    """Detect the actual image format from file content and return format info."""
    with open(image_path, "rb") as f:
        header = f.read(16)  # Read more bytes for better detection
    
    # Check for HTML files (common issue)
    if header.startswith(b'<!DOCTYPE') or header.startswith(b'<html'):
        raise Exception(f"File appears to be HTML, not an image: {image_path}")
    
    # Check file signatures
    if header.startswith(b'\xff\xd8\xff'):
        return 'jpeg', False  # format, needs_conversion
    elif header.startswith(b'\x89PNG\r\n\x1a\n'):
        return 'png', False
    elif header.startswith(b'GIF87a') or header.startswith(b'GIF89a'):
        return 'gif', False
    elif header.startswith(b'RIFF') and header[8:12] == b'WEBP':
        return 'webp', False
    elif len(header) >= 12 and header[4:12] == b'ftypavif':
        return 'jpeg', True  # Convert AVIF to JPEG
    else:
        # Check if it might be a text file
        try:
            header_str = header.decode('utf-8', errors='ignore')
            if any(html_tag in header_str.lower() for html_tag in ['<html', '<!doctype', '<head', '<body']):
                raise Exception(f"File appears to be HTML/text, not an image: {image_path}")
        except:
            pass
        
        # Try to open with PIL to see if it's a valid image
        try:
            with Image.open(image_path) as img:
                # If PIL can open it, convert to JPEG for safety
                return 'jpeg', True
        except Exception:
            raise Exception(f"Unsupported or corrupted image file: {image_path}")

def convert_to_jpeg_bytes(image_path):
    """Convert any image format to JPEG bytes."""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if necessary (for formats like PNG with transparency)
            if img.mode in ('RGBA', 'LA', 'P'):
                # Create white background for transparent images
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save as JPEG to BytesIO
            jpeg_buffer = BytesIO()
            img.save(jpeg_buffer, format='JPEG', quality=95)
            return jpeg_buffer.getvalue()
    except Exception as e:
        raise Exception(f"Failed to convert image to JPEG: {str(e)}")

def encode_image_to_base64(image_path, convert_to_jpeg=False):
    """Encode an image file to base64 string."""
    if convert_to_jpeg:
        jpeg_bytes = convert_to_jpeg_bytes(image_path)
        return base64.b64encode(jpeg_bytes).decode('utf-8')
    else:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

def encode_image_from_url(image_url):
    """Download and encode an image from URL to base64 string."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(image_url, timeout=30, headers=headers)
        response.raise_for_status()
        return base64.b64encode(response.content).decode('utf-8'), response.headers.get('content-type', '')
    except Exception as e:
        raise Exception(f"Failed to download image from URL: {str(e)}")

def img_tagging(image_input, prompt=None, region="us-west-2", model_id="us.amazon.nova-pro-v1:0", 
                aws_access_key_id=None, aws_secret_access_key=None, return_metrics=False, use_cache=True):
    """
    Image tagging function that works with both local files and URLs
    
    Args:
        image_input: Either a local file path or URL to the image
        prompt: Custom user prompt (optional, uses default if None)
        region: AWS region
        model_id: Nova model ID
        aws_access_key_id: AWS access key (optional)
        aws_secret_access_key: AWS secret key (optional)
        return_metrics: If True, returns (text, metrics) tuple instead of just text
        use_cache: If True, enables prompt caching for system prompt (default: True)
    
    Returns:
        str or tuple: The generated text response from the model, or (text, metrics) if return_metrics=True
    """
    # System prompt from novaImageAnalysis-1.py
    system_prompt = """
    ##ROLE##
You are an advanced image classification specialist analyzing e-commerce product images to identify multiple relevant category labels from the ##REFERENCE_CATEGORIES##. Your primary focus is comprehensive coverage (recall) over strict precision.

##PRIMARY_TASK##
Analyze the provided image and identify **UP TO 5 MOST RELEVANT** category labels from ##REFERENCE_CATEGORIES##. You are **STRONGLY ENCOURAGED** to output multiple labels when the image could reasonably belong to several categories or when there's reasonable uncertainty.

##CRITICAL_MULTI_LABEL_REQUIREMENTS##

**IMPORTANT: Multiple Labels Are Expected and Preferred**
- **Output 1-3 labels** whenever multiple categories from ##REFERENCE_CATEGORIES## are plausible
- Think of this as providing comprehensive classification alternatives, not just the single best match
- Consider related categories, different levels of specificity, and various product aspects
- **Even if one label seems most accurate, include other reasonable alternatives**
- **Err on the side of inclusion** - it's better to flag potential matches than miss restricted items

**When to Output Multiple Labels (ENCOURAGED)**
- The product could belong to multiple related categories from ##REFERENCE_CATEGORIES##
- Different components or features suggest different classifications  
- Similar items with slight variations might fit different categories
- There's reasonable uncertainty between closely related categories
- Borderline cases that might pose compliance risks
- Items that share characteristics with multiple restricted categories

##ANALYSIS_FRAMEWORK##
Follow this enhanced 4-step process using chain-of-thought reasoning:

**Step 1: Comprehensive Physical Attribute Extraction**
- Extract **ALL** objective physical attributes of **ALL** items in the image
- Focus on neutral, geometric, and material properties
- **DO NOT** interpret purpose or function yet - just observe
- **Examples of attributes to extract:**
  - **Shape:** Cylindrical, rectangular, pointed tip, curved edge, ring-shaped, telescoping
  - **Components:** Handle, blade, trigger, nozzle, switch, USB port, chain, electrodes
  - **Material/Texture:** Metallic, plastic, fabric, wooden, smooth, serrated, crystalline
  - **Text/Markings:** Any visible letters, numbers, symbols, warnings, brand names
  - **Size indicators:** Dimensions, child-oriented designs, scale references

**Step 2: Multi-Aspect Feature Analysis**
Based on your description, identify **ALL** potentially relevant features:
- **Primary features:** Main identifying characteristics
- **Secondary features:** Additional elements that might suggest other categories
- **Contextual clues:** Usage scenarios, accompanying items, packaging
- **Safety indicators:** Warning labels, hazard symbols, age restrictions
- **Design intent:** Child-friendly designs, tactical appearances, concealment features

**Step 3: Comprehensive Category Matching**
Compare identified features with **ALL** categories in ##REFERENCE_CATEGORIES##:
- **Match observed features** with the "Key Features" listed for each category
- **Consider multiple perspectives** - how different reviewers might categorize this item
- **Include borderline cases** that might pose compliance risks
- **Don't dismiss close matches** - include categories that are "reasonably close"
- **Think about related categories** that share similar characteristics
- **Consider both obvious and subtle matches**

**Step 4: Multi-Label Classification Output**
Provide classification following these **enhanced rules**:
- **PREFERRED: Output 2-5 category names** when multiple reasonable matches exist
- **Include borderline cases** that human reviewers should evaluate
- **Order by confidence level** (highest confidence first)
- **Use ONLY the OUTPUT_LABEL values** from ##REFERENCE_CATEGORIES## for all matches
- Separate multiple categories with English commas "," only
- **Single label output should be exceptional** - only when truly only one category applies
- If absolutely no categories match, return "无" (this should be rare)
- Format: {"result":"category1,category2,category3"}

##OUTPUT_REQUIREMENTS##
- **Strongly favor multi-label outputs** - comprehensive coverage is key
- **Include "reasonable doubt" cases** - let human reviewers make final decisions  
- **Think broadly** about the image and provide multiple classification perspectives
- **Use ONLY the OUTPUT_LABEL values from ##REFERENCE_CATEGORIES##** for classification results
- Follow exact JSON format: {"result":"category1,category2"}
- **When uncertain between categories, include both rather than choosing one**
- **DO NOT include escaped Unicode characters** in output

##COMPLIANCE_GUIDELINES##
**Critical Guidelines for E-commerce Compliance:**
- **Prioritize recall over precision** - better to flag potential issues than miss them
- **Include items that could reasonably match** definitions from ##REFERENCE_CATEGORIES##, even with uncertainty
- **Borderline cases should be included** rather than excluded
- **Multiple perspectives benefit compliance** - provide comprehensive options
- **Focus on identifying potential compliance risks** across all applicable categories
- **Consider how different reviewers might interpret** the same image
- **Remember: Your role is to catch potential issues, not make final judgments**

##EXECUTION_INSTRUCTIONS##
You must strictly follow ##ANALYSIS_FRAMEWORK## and ##OUTPUT_REQUIREMENTS##. 
- Follow the 4-step analysis framework using chain-of-thought reasoning, then provide your response directly in the required JSON format without showing your analysis steps
- Refer to ##REFERENCE_CATEGORIES## for all classification decisions
- Apply ##COMPLIANCE_GUIDELINES## throughout your analysis

    ## Reference Categories

# Categorized Product Restrictions

## Automotive and Safety Equipment

# Reformatted Product Restrictions

## Automotive and Safety Equipment

1. **Steering Wheel Phone Mount**
   - **Definition**: Phone mounts designed to clip onto steering wheels for video viewing while driving, violating safety laws and potentially obstructing airbags.
   - **OUTPUT_LABEL**: 方向盘手机支架

2. **Car Seatbelt Clip/Unbuckler**
   - **Definition**: Devices inserted into seatbelt buckles to disable safety warnings without actually wearing the seatbelt, bypassing vehicle safety systems.
   - **OUTPUT_LABEL**: 汽车安全带插扣/脱扣器

3. **Car Steering Wheel Stickers**
   - **Definition**: Adhesive stickers or decals designed for application on steering wheels that may interfere with safety systems.
   - **OUTPUT_LABEL**: 汽车方向盘贴

4. **License Plate Obstruction Devices**
   - **Definition**: Remote-controlled devices with motorized frames that can open/close to conceal license plates from view.
   - **OUTPUT_LABEL**: 汽车车牌遮挡

## Weapons and Combat Items

### Knives and Blades

5. **Dagger**
   - **Definition**: Short-bladed weapons with sharp pointed tips designed primarily for thrusting attacks, featuring compact design and dual-edge construction.
   - **OUTPUT_LABEL**: 匕首

6. **Knives**
   - **Definition**: Sharp-bladed cutting instruments including utility, tactical, folding, or ornamental types with various handle designs and slashing capability.
   - **OUTPUT_LABEL**: 刀具

7. **Oversized Knives**
   - **Definition**: Knives with blade lengths exceeding 15cm (5.9 inches), providing enhanced cutting capability beyond standard utility knives.
   - **OUTPUT_LABEL**: 超长刀具

8. **Switchblade**
   - **Definition**: Folding knives with spring-deployed blades activated by buttons or levers, featuring concealed blades and rapid deployment mechanisms.
   - **OUTPUT_LABEL**: 弹簧刀

9. **Butterfly Knife**
   - **Definition**: Folding knives with dual rotating handles that conceal and deploy the blade, featuring pivoting handles and latch mechanisms.
   - **OUTPUT_LABEL**: 蝴蝶刀

10. **Bastinelli Mako Shark Knife**
    - **Definition**: Aggressively designed knives with distinctive shark-inspired profiles and enhanced cutting capabilities from the Bastinelli brand.
    - **OUTPUT_LABEL**: 灰鲭鲨刀

11. **Disguised Knives**
    - **Definition**: Sharp blades concealed within everyday objects like pens, keychains, or tools, featuring hidden deployment mechanisms.
    - **OUTPUT_LABEL**: 伪装刀具

12. **Claw Knife**
    - **Definition**: Curved knives with finger rings or holes in handles for enhanced grip and slashing capability, worn on hands.
    - **OUTPUT_LABEL**: 爪刀

### Projectile Weapons

13. **Slingshot**
    - **Definition**: Handheld Y-shaped projectile weapons using elastic bands and pouches to launch stones or pellets at targets.
    - **OUTPUT_LABEL**: 弹弓

14. **Slingshot Crossbow**
    - **Definition**: Hybrid weapons combining crossbow-style frames with elastic propulsion systems and trigger mechanisms for projectile launching.
    - **OUTPUT_LABEL**: 弹弓弩

15. **Throwing Darts**
    - **Definition**: Sharp, weighted projectiles designed for throwing at targets, featuring balanced aerodynamic shapes and pointed tips.
    - **OUTPUT_LABEL**: 飞镖

16. **Bow**
    - **Definition**: Traditional archery weapons using curved frames and bowstrings with tension to launch arrows at targets.
    - **OUTPUT_LABEL**: 弓

17. **Arrows/Arrowheads**
    - **Definition**: Projectiles designed for bow use, featuring sharp points, shafts, and fletching for stabilization during flight.
    - **OUTPUT_LABEL**: 箭/箭头

18. **Crossbow**
    - **Definition**: Horizontal bow assemblies mounted on stocks with trigger mechanisms for launching bolts with enhanced accuracy.
    - **OUTPUT_LABEL**: 弩

### Impact Weapons

19. **Nunchucks**
    - **Definition**: Traditional martial arts weapons consisting of two rigid sticks connected by chains or ropes for swinging combat.
    - **OUTPUT_LABEL**: 双截棍

20. **Disguised Toothpicks**
    - **Definition**: Sharp pointed objects disguised as innocent toothpicks but designed as concealed weapons for stabbing.
    - **OUTPUT_LABEL**: 伪装牙签

21. **Key Stick**
    - **Definition**: Rigid weapons disguised as or incorporated into keychain accessories, designed for impact and stabbing attacks.
    - **OUTPUT_LABEL**: 钥匙棍

### Firearms-Related Items

22. **Replica Firearms**
    - **Definition**: Non-functional firearm replicas with realistic appearance and firearm silhouettes but no shooting capability.
    - **OUTPUT_LABEL**: 仿真枪

23. **Replica Bullets**
    - **Definition**: Non-functional bullet replicas with metallic appearance and bullet shapes, often used as keychains or jewelry.
    - **OUTPUT_LABEL**: 仿真子弹

24. **Magazine**
    - **Definition**: Ammunition feeding devices with spring-loaded mechanisms and specific firearm compatibility for cartridge storage.
    - **OUTPUT_LABEL**: 弹匣

25. **Magazine Loader**
    - **Definition**: Tools designed to assist in loading ammunition into magazines, featuring loading assistance mechanisms and magazine compatibility.
    - **OUTPUT_LABEL**: 弹匣装弹器

26. **Scope**
    - **Definition**: Optical sighting devices for firearms featuring magnification lenses, crosshairs, and mounting systems for accuracy.
    - **OUTPUT_LABEL**: 瞄准镜

27. **Gun Barrel**
    - **Definition**: Cylindrical tubes through which projectiles are fired, featuring rifling grooves and muzzle openings for bullet guidance.
    - **OUTPUT_LABEL**: 枪管

28. **Gun Stock**
    - **Definition**: Rear portions of firearms providing shoulder support and attachment points for firearm mechanisms.
    - **OUTPUT_LABEL**: 枪托

29. **Firearm Rail System**
    - **Definition**: Standardized mounting systems for firearm accessories featuring rail designs and accessory attachment capability.
    - **OUTPUT_LABEL**: 枪支导轨

30. **Gun Grip**
    - **Definition**: Hand-holding portions of firearms with ergonomic designs and textured surfaces for secure weapon control.
    - **OUTPUT_LABEL**: 枪支握把

31. **Silencer/Suppressor**
    - **Definition**: Cylindrical devices with threaded attachments and sound dampening chambers designed to reduce firearm muzzle noise.
    - **OUTPUT_LABEL**: 消音器

### Law Enforcement Equipment

32. **Police Baton/Expandable Baton**
    - **Definition**: High-strength metal batons with telescoping designs, aggressive protrusions, and police markings for law enforcement use.
    - **OUTPUT_LABEL**: 警棍/甩棍

33. **Police Spray**
    - **Definition**: Defensive sprays in pressurized canisters including pepper spray, tear gas, and anti-wolf sprays with directional nozzles.
    - **OUTPUT_LABEL**: 警用喷雾

34. **Handcuffs/Thumb Cuffs/Leg Irons**
    - **Definition**: Metal restraint devices with locking mechanisms meeting specific size criteria for restricting human movement.
    - **OUTPUT_LABEL**: 手铐、拇指烤、脚镣

### Electric Weapons

35. **Stun Gun/Electric Baton**
    - **Definition**: Weapons with exposed electrodes producing high-voltage electric shocks, featuring aggressive protrusions and electric arc generation.
    - **OUTPUT_LABEL**: 电击器/电击棒

36. **Flashlight with Stun Function**
    - **Definition**: Dual-purpose devices appearing as flashlights but containing hidden electrodes for electric shock capability.
    - **OUTPUT_LABEL**: 带电击功能的手电筒

### Hand Weapons

37. **Knuckle Duster/Iron Lotus**
    - **Definition**: Hand-worn offensive weapons with finger holes and protruding elements designed to enhance fist-based attacks.
    - **OUTPUT_LABEL**: 指虎铁莲花

## Hazardous and Explosive Materials

### Fire and Ignition Sources

38. **Gas-Free Lighter**
    - **Definition**: Electronic ignition devices without gas reservoirs, using piezoelectric or battery-powered spark generation for lighting.
    - **OUTPUT_LABEL**: 打火器（无燃气）

39. **Electronic Lighter**
    - **Definition**: Lighters using piezoelectric ignition systems with rechargeable capability and electrical spark electrodes instead of flames.
    - **OUTPUT_LABEL**: 电子打火器

40. **Gas Lighter**
    - **Definition**: Traditional lighters powered by combustible gas with reservoirs, flame adjustment controls, and ignition mechanisms.
    - **OUTPUT_LABEL**: 燃气打火机

41. **Flint/Fire Starter**
    - **Definition**: Traditional fire-starting tools using flint materials and striker tools to generate sparks for ignition.
    - **OUTPUT_LABEL**: 打火石、打火棒

42. **Solid Alcohol**
    - **Definition**: Wax-like flammable solid fuel blocks used for heating and cooking applications as portable fuel sources.
    - **OUTPUT_LABEL**: 固体酒精

43. **Red/White Phosphorus**
    - **Definition**: Highly reactive crystalline phosphorus compounds in powder form with extreme reactivity and hazardous fire properties.
    - **OUTPUT_LABEL**: 红磷、白磷

44. **Matches**
    - **Definition**: Traditional ignition sticks with wooden or paper bodies and combustible tips requiring striking surfaces.
    - **OUTPUT_LABEL**: 火柴

45. **Magnesium Powder**
    - **Definition**: Fine silvery metallic powder with intense burning capability and high flammability for fire-starting applications.
    - **OUTPUT_LABEL**: 镁粉

### Explosive Materials

46. **Gunpowder**
    - **Definition**: Explosive granular powder used as chemical propellant with explosive capability for ammunition and fireworks.
    - **OUTPUT_LABEL**: 火药

47. **Nail Gun Cartridge**
    - **Definition**: Explosive cartridges designed for construction nail guns, containing explosive charges for driving nails.
    - **OUTPUT_LABEL**: 射钉弹

48. **Fireworks and Firecrackers**
    - **Definition**: Pyrotechnic devices with explosive compositions producing colorful visual and audio effects for celebrations.
    - **OUTPUT_LABEL**: 烟花爆竹

49. **Explosives**
    - **Definition**: High-explosive materials for demolition and blasting with dangerous compositions and high explosive capability.
    - **OUTPUT_LABEL**: 炸药

### Smoke and Gas Devices

50. **Sky Lantern**
    - **Definition**: Traditional floating paper or silk lanterns with open flame heat sources and uncontrolled flight paths.
    - **OUTPUT_LABEL**: 孔明灯

51. **Fire Extinguisher**
    - **Definition**: Portable firefighting devices with pressurized vessels, spray nozzles, and fire suppression agents.
    - **OUTPUT_LABEL**: 灭火器

52. **Smoke Bomb**
    - **Definition**: Compact devices producing colored smoke through ignition mechanisms for signaling or special effects.
    - **OUTPUT_LABEL**: 烟雾弹

53. **Smoke Cake/Disc**
    - **Definition**: Flat disc-shaped smoke-producing devices with ignition points for effects or ceremonial rituals.
    - **OUTPUT_LABEL**: 烟饼（片）

54. **Relight Candles**
    - **Definition**: Novelty candles with self-igniting mechanisms that automatically relight after being blown out.
    - **OUTPUT_LABEL**: 重燃蜡烛

### Pressurized Containers

55. **Cassette Stove**
    - **Definition**: Portable cooking stoves using pressurized gas cartridges with burner assemblies for outdoor cooking.
    - **OUTPUT_LABEL**: 卡式炉

56. **Confetti Cannon**
    - **Definition**: Non-combustible cylindrical tubes ejecting confetti or streamers through mechanical action without fire or explosives.
    - **OUTPUT_LABEL**: 礼花筒

57. **Gas Canister**
    - **Definition**: Pressurized metal containers with pressure valves containing flammable or compressed gases.
    - **OUTPUT_LABEL**: 气罐

58. **Aerosol Spray**
    - **Definition**: Pressurized canisters with spray nozzles and aerosol delivery systems dispensing various substances.
    - **OUTPUT_LABEL**: 压罐喷雾

59. **Kerosene**
    - **Definition**: Flammable liquid hydrocarbon fuel with high flammability used for heating, lighting, and fuel applications.
    - **OUTPUT_LABEL**: 煤油

## Controlled Substances and Related Items

### Drugs and Drug Precursors

60. **Drugs**
    - **Definition**: Natural and synthetic controlled substances including cannabis, opiates, cocaine, and synthetic drugs with psychoactive properties.
    - **OUTPUT_LABEL**: 毒品

61. **Drug Paraphernalia**
    - **Definition**: Items directly associated with drug consumption including pipes, syringes, and preparation tools with drug residue.
    - **OUTPUT_LABEL**: 吸毒工具

62. **Precursor Chemicals**
    - **Definition**: Regulated chemicals used in drug manufacturing including ephedrine, acetone, and various solvents with manufacturing capability.
    - **OUTPUT_LABEL**: 易制毒成分

63. **Drug Manufacturing Tools - Capsule Filling Machine**
    - **Definition**: Equipment for filling capsules with powdered substances, potentially used in illegal drug production and distribution.
    - **OUTPUT_LABEL**: 制毒工具-胶囊填充机

### Smoking and Tobacco-Related Items

64. **Illegal Water Pipe**
    - **Definition**: Water pipes modified or designed for illegal substance consumption, showing drug residue or non-standard heating elements.
    - **OUTPUT_LABEL**: 非法水烟壶

65. **Illegal Smoking Pipe**
    - **Definition**: Smoking pipes with evidence of illegal substance use, featuring drug residue or structural modifications.
    - **OUTPUT_LABEL**: 非法烟斗

66. **E-cigarettes and Accessories**
    - **Definition**: Electronic vaping devices with battery systems, atomizers, and e-liquid compatibility for nicotine delivery.
    - **OUTPUT_LABEL**: 电子烟及配件

67. **Tobacco/Tobacco Shreds**
    - **Definition**: Processed tobacco products and smoking materials with nicotine content prepared for smoking consumption.
    - **OUTPUT_LABEL**: 烟草烟丝

68. **Arabian Water Pipe and Accessories**
    - **Definition**: Traditional water pipes with proper certifications designed exclusively for tobacco use with traditional craftsmanship.
    - **OUTPUT_LABEL**: 阿拉伯水烟壶及配件

69. **Cigarette Rolling Tools**
    - **Definition**: Hand-rolling tools including rolling machines and accessories for forming cigarettes from loose tobacco.
    - **OUTPUT_LABEL**: 卷烟器具

70. **Rolling Papers/Cigarettes**
    - **Definition**: Thin papers designed for tobacco rolling and finished tobacco products with standard dimensions.
    - **OUTPUT_LABEL**: 卷烟纸/烟卷

71. **Legal Smoking Pipes and Accessories**
    - **Definition**: Traditional pipes made from natural materials for tobacco use only, featuring traditional craftsmanship.
    - **OUTPUT_LABEL**: 可售烟斗及配件

72. **Hookah Charcoal**
    - **Definition**: Specialized plant-based charcoal for water pipe heating with high-temperature burning and long duration.
    - **OUTPUT_LABEL**: 水烟碳

73. **Cigar Accessories**
    - **Definition**: Premium tools for cigar preparation including cutters, punches, and accessories made from quality materials.
    - **OUTPUT_LABEL**: 雪茄配件

74. **Tobacco Grinder**
    - **Definition**: Mechanical tools with grinding mechanisms for processing tobacco into uniform particles with size control.
    - **OUTPUT_LABEL**: 烟草研磨器

75. **Cigarette Case/Ashtray**
    - **Definition**: Accessories for cigarette storage and ash disposal featuring smoking-related imagery and functional design.
    - **OUTPUT_LABEL**: 烟盒/烟灰缸

76. **Cigarette Holder and Accessories**
    - **Definition**: Holders for cigarettes with filtering capability and cleaning accessories for maintenance and use.
    - **OUTPUT_LABEL**: 烟嘴及配件

77. **Tobacco-Related Imagery Prohibited**
    - **Definition**: Products displaying smoking actions, cigarettes, or tobacco-related graphics and imagery that promote smoking.
    - **OUTPUT_LABEL**: 烟草图案形状禁上mx

### Gambling Items

78. **Gambling Chips**
    - **Definition**: Circular or square gaming tokens with face value markings and casino compatibility for monetary gambling.
    - **OUTPUT_LABEL**: 赌博用筹码

## Medical and Health-Related Items

### Medical Devices and Instruments

79. **Syringes**
    - **Definition**: Medical injection devices with barrel and plunger assemblies, needle attachments, and liquid injection capability.
    - **OUTPUT_LABEL**: 针管

80. **Comedone Extractor**
    - **Definition**: Dermatological tools with sharp pointed tips and extraction loops for removing blackheads and pimples.
    - **OUTPUT_LABEL**: 粉刺针

81. **Stethoscope**
    - **Definition**: Medical instruments with chest pieces, tubing, and earpieces for listening to internal body sounds.
    - **OUTPUT_LABEL**: 听诊器

82. **Derma Roller**
    - **Definition**: Cosmetic devices with roller mechanisms and multiple micro-needles for skin penetration and treatment.
    - **OUTPUT_LABEL**: 微针滚轮

83. **Ear Piercing Gun**
    - **Definition**: Spring-loaded devices with sterile needle/stud assemblies for creating ear piercings safely.
    - **OUTPUT_LABEL**: 穿耳器

84. **Digital Thermometer**
    - **Definition**: Electronic temperature-measuring devices with digital displays and medical accuracy for body temperature monitoring.
    - **OUTPUT_LABEL**: 数字体温计

85. **Baby Nasal Aspirator**
    - **Definition**: Suction devices with infant-safe designs for clearing nasal congestion in babies and toddlers.
    - **OUTPUT_LABEL**: 婴儿洗鼻器

86. **Electric Toothbrush**
    - **Definition**: Battery-powered toothbrushes with motor mechanisms, oscillating/vibrating heads, and charging capability.
    - **OUTPUT_LABEL**: 电动牙刷

87. **Tongue Scraper**
    - **Definition**: Oral hygiene tools with scraping surfaces and ergonomic handles for tongue cleaning.
    - **OUTPUT_LABEL**: 舌刮

88. **Water Flosser**
    - **Definition**: Dental cleaning devices using water pressure with reservoirs, pumps, and cleaning nozzles.
    - **OUTPUT_LABEL**: 冲牙器

89. **Dental Mirror**
    - **Definition**: Small angled mirrors with reflecting surfaces designed for dental examination and oral inspection.
    - **OUTPUT_LABEL**: 牙镜

90. **Dental Forceps**
    - **Definition**: Precision instruments with gripping ends for dental procedures requiring precise control and manipulation.
    - **OUTPUT_LABEL**: 牙镊

91. **Electronic Ear Pick**
    - **Definition**: Digital ear cleaning devices with camera integration and LED lighting for safe ear cleaning.
    - **OUTPUT_LABEL**: 电子挖耳勺

92. **Breast Pump**
    - **Definition**: Devices with suction mechanisms for expressing breast milk, designed for maternal health applications.
    - **OUTPUT_LABEL**: 吸奶器

93. **Blood Glucose Meter**
    - **Definition**: Devices with test strip compatibility and digital readouts for measuring blood sugar levels.
    - **OUTPUT_LABEL**: 血糖仪

94. **Blood Pressure Monitor**
    - **Definition**: Cardiovascular monitoring devices with inflatable cuffs and pressure gauges for measuring arterial blood pressure.
    - **OUTPUT_LABEL**: 血压仪/血压计

95. **Pulse Oximeter**
    - **Definition**: Finger clip devices with LED sensors for measuring blood oxygen saturation levels.
    - **OUTPUT_LABEL**: 血氧仪

96. **Cupping Set**
    - **Definition**: Traditional therapy devices with cup-shaped vessels and suction mechanisms for therapeutic skin treatment.
    - **OUTPUT_LABEL**: 拔罐器

97. **Bedside Rail**
    - **Definition**: Support rails with sturdy construction and bed attachment systems for mobility assistance and fall prevention.
    - **OUTPUT_LABEL**: 床边扶手

### Beauty and Cosmetic Devices

98. **EMS Microcurrent Beauty Device**
    - **Definition**: Electronic facial treatment devices using microcurrent generation for beauty treatments and anti-aging applications.
    - **OUTPUT_LABEL**: EMS微电流美容仪/颈纹仪/电子脸部滚轮

99. **Laser Hair Removal Device**
    - **Definition**: Consumer laser devices with hair follicle targeting capability for permanent hair reduction treatments.
    - **OUTPUT_LABEL**: 激光脱毛仪

100. **LED Face Mask**
     - **Definition**: Light therapy masks with LED arrays providing facial coverage for skincare treatment applications.
     - **OUTPUT_LABEL**: LED面罩

101. **Iontophoresis Device**
     - **Definition**: Skincare devices using electrical current generation for ionic delivery and product penetration enhancement.
     - **OUTPUT_LABEL**: 离子导入导出仪器

102. **Electrotherapy Device**
     - **Definition**: Therapeutic devices providing electrical pulse generation for muscle stimulation and therapeutic applications.
     - **OUTPUT_LABEL**: 电疗仪

103. **LED Teeth Whitening Light**
     - **Definition**: LED light-emitting devices for dental applications designed to enhance teeth whitening procedures.
     - **OUTPUT_LABEL**: LED牙齿美白灯

### Health Monitoring and Respiratory Devices

104. **Mouth Breathing Strips**
     - **Definition**: Adhesive strips applied to lips to promote nasal breathing during sleep and modify breathing patterns.
     - **OUTPUT_LABEL**: 止鼾贴（贴在嘴巴上)/呼吸贴

105. **Anti-Snoring Device**
     - **Definition**: Devices designed to reduce or eliminate snoring through airway modification and sleep improvement.
     - **OUTPUT_LABEL**: 止鼾器

106. **Fever Reducing Patch**
     - **Definition**: Cooling patches with gel and adhesive backing for temperature reduction in children and adults.
     - **OUTPUT_LABEL**: 退热贴

### Personal Hygiene and Feminine Care

107. **Feminine Hygiene Products - Tampons**
     - **Definition**: Absorbent products with internal insertion design for menstrual protection and feminine hygiene.
     - **OUTPUT_LABEL**: 女性卫生用品 棉条

108. **Sanitary Pads**
     - **Definition**: External feminine hygiene products with absorbent cores and adhesive backing for menstrual protection.
     - **OUTPUT_LABEL**: 卫生巾

109. **Menstrual Cup**
     - **Definition**: Reusable silicone cups with cup shapes designed for menstrual fluid collection and eco-friendly protection.
     - **OUTPUT_LABEL**: 月经杯

### Contact Lenses and Vision Care

110. **Colored Contact Lenses**
     - **Definition**: Cosmetic contact lenses for eye color enhancement with optional vision correction for appearance modification.
     - **OUTPUT_LABEL**: 美瞳

111. **Contact Lens Case**
     - **Definition**: Storage containers with dual compartments and solution compatibility for contact lens storage and care.
     - **OUTPUT_LABEL**: 隐形眼镜盒

### Medical Tattooing and Body Modification

112. **Tattoo Equipment**
     - **Definition**: Professional tattooing equipment including needle assemblies, motorized machines, and ink delivery systems.
     - **OUTPUT_LABEL**: 纹身针 纹身套装 纹身枪仪器 纹身喷嘴 纹身握把

### Health and Wellness Products

113. **Adhesive Bandages**
     - **Definition**: Small bandages with adhesive backing and sterile pads for minor wound protection and first aid.
     - **OUTPUT_LABEL**: 创可贴

114. **Weight Loss Patches**
     - **Definition**: Transdermal adhesive patches claiming weight loss benefits through skin application and chemical delivery.
     - **OUTPUT_LABEL**: 减肥贴

115. **Pharmaceuticals**
     - **Definition**: All medications including pills, tablets, and medicinal ointments with active pharmaceutical ingredients and therapeutic purposes.
     - **OUTPUT_LABEL**: 药品

## Electronics and Technology

### Power and Battery Products

116. **Power Bank**
     - **Definition**: Portable battery devices with lithium cores, USB ports, and charging capability for electronic equipment.
     - **OUTPUT_LABEL**: 充电宝

117. **Pure Lithium Batteries**
     - **Definition**: Standalone lithium batteries with CR/BR designations, lithium composition, and primary cell design.
     - **OUTPUT_LABEL**: 纯锂电池

118. **Button Batteries**
     - **Definition**: Small coin-shaped batteries with compact size and various voltage ratings for electronic devices.
     - **OUTPUT_LABEL**: 纽扣电池

119. **Button Battery Powered Products**
     - **Definition**: Small electronic devices with button battery compartments and battery dependency for operation.
     - **OUTPUT_LABEL**: 纽扣电池供电的产品

### Communication and Surveillance Equipment

120. **Walkie-Talkies and Accessories**
     - **Definition**: Two-way radio communication devices with handheld design and specialized components for radio transmission.
     - **OUTPUT_LABEL**: 对讲机及配件

121. **Professional Gas Masks**
     - **Definition**: Professional-grade respiratory protection equipment with full/half face coverage and replaceable filter systems.
     - **OUTPUT_LABEL**: 专业防毒面罩

122. **TV Set-Top Box**
     - **Definition**: Signal reception devices for converting television signals with format conversion and TV connectivity.
     - **OUTPUT_LABEL**: 电视机顶盒

123. **Smart Doorbell**
     - **Definition**: Connected doorbells with wireless connectivity, camera integration, and smartphone app compatibility.
     - **OUTPUT_LABEL**: 智能门铃

### Electrical Classification

124. **High Voltage Electrical Products**
     - **Definition**: Products requiring direct connection to high-voltage electrical circuits with power adapters included.
     - **OUTPUT_LABEL**: 强电类商品

125. **Low Voltage Electrical Products**
     - **Definition**: Devices operating on low voltage through battery, USB, solar charging, or low voltage design.
     - **OUTPUT_LABEL**: 弱电类商品

126. **Wireless Enabled Products**
     - **Definition**: Devices with wireless communication capabilities including WiFi/Bluetooth connectivity and network capability.
     - **OUTPUT_LABEL**: 无线功能类商品

127. **Disinfection/Pest Control/Purification Products**
     - **Definition**: Products with active treatment agents, purification mechanisms, and sanitizing or pest elimination capabilities.
     - **OUTPUT_LABEL**: 消毒、灭虫、净化功能

### Personal Electronics

128. **Mobile Phones**
     - **Definition**: Cellular communication devices with network compatibility, touchscreen interfaces, and voice/data transmission capability.
     - **OUTPUT_LABEL**: 手机

129. **Non-GFCI Hair Dryers/Hot Air Brushes**
     - **Definition**: Hair styling devices with heating elements and air circulation but lacking ground fault circuit protection.
     - **OUTPUT_LABEL**: 无漏保吹风机/热风梳

## Surveillance and Optical Equipment

130. **Night Vision Binoculars**
     - **Definition**: Optical devices with light amplification and binocular design for enhanced vision in low-light conditions.
     - **OUTPUT_LABEL**: 夜视望远镜

131. **Laser Pointer**
     - **Definition**: Handheld devices with pen-like design emitting focused laser beams for pointing applications.
     - **OUTPUT_LABEL**: 激光笔

132. **Toy Drones**
     - **Definition**: Recreational unmanned aircraft with remote control and flight capability for entertainment purposes.
     - **OUTPUT_LABEL**: 玩具无人机

133. **Consumer Drones**
     - **Definition**: Advanced unmanned aircraft with camera integration, GPS capability, and advanced flight controls.
     - **OUTPUT_LABEL**: 消费级无人机

## Transportation and Mobility

134. **Electric Scooter**
     - **Definition**: Electric-powered personal transportation devices with motors, standing platforms, and handlebar steering.
     - **OUTPUT_LABEL**: 电动滑板车

135. **Electric Bicycle**
     - **Definition**: Bicycles with electric motor assistance, pedaling capability, and bicycle frame design for transportation.
     - **OUTPUT_LABEL**: 电动自行车

136. **Self-Balancing Scooter**
     - **Definition**: Two-wheeled personal transporters with self-balancing technology, foot platforms, and gyroscopic control.
     - **OUTPUT_LABEL**: 平衡车

## Animal and Pet Products

### Pet Care and Feeding

137. **Pet Hair Dye**
     - **Definition**: Chemical hair coloring products with pet-specific formulation for changing pet hair color appearance.
     - **OUTPUT_LABEL**: 宠物染发剂

138. **Other Pet Food - Except Cat/Dog/Horse/Fish**
     - **Definition**: Food products with species-specific nutrition for pets other than cats, dogs, horses, and fish.
     - **OUTPUT_LABEL**: 其他宠物食品（除猫/狗/马/鱼）

139. **General Pet Food - Cat/Dog/Horse/Fish**
     - **Definition**: Standard pet food products with species-appropriate nutrition for common domesticated animals.
     - **OUTPUT_LABEL**: 一般宠物食品（猫狗马鱼）

140. **Pet Medications**
     - **Definition**: Pharmaceutical products with veterinary formulation, animal-specific dosing, and therapeutic purpose for pets.
     - **OUTPUT_LABEL**: 宠物药品

141. **Pest Control Collar**
     - **Definition**: Collar-worn devices with pest repellent agents and extended wear capability for pet pest control.
     - **OUTPUT_LABEL**: 驱虫项圈

## Chemical and Environmental Products

### Pest Control and Chemicals

142. **Animal Trapping/Killing Tools**
     - **Definition**: Devices with trap mechanisms and lethal components designed to capture or kill wild animals.
     - **OUTPUT_LABEL**: 动物捕杀工具

143. **Soil**
     - **Definition**: Natural or processed earth materials with organic matter content and nutrient composition for gardening.
     - **OUTPUT_LABEL**: 土壤

144. **Dried Plants**
     - **Definition**: Dehydrated plant materials with preserved plant matter and reduced moisture content for extended shelf life.
     - **OUTPUT_LABEL**: 干植物

145. **Live Plants**
     - **Definition**: Living plant specimens including seedlings and mature plants with growth potential requiring care.
     - **OUTPUT_LABEL**: 植物活体

146. **Seeds**
     - **Definition**: Plant reproductive units with germination potential, species identification, and planting capability for growing.
     - **OUTPUT_LABEL**: 种子

147. **Chemical Mothballs**
     - **Definition**: Chemical pest deterrents containing naphthalene or paradichlorobenzene with volatile compounds and pest deterrent properties.
     - **OUTPUT_LABEL**: 化学樟脑丸

148. **Approved Insecticide/Rodenticide**
     - **Definition**: Legally permitted pest control products with active pest control agents and regulatory approval for targeted elimination.
     - **OUTPUT_LABEL**: 可售杀虫杀鼠

149. **Mosquito Coils**
     - **Definition**: Spiral coil designs with slow combustion and mosquito deterrent compounds releasing insect repellent smoke.
     - **OUTPUT_LABEL**: 蚊香

150. **Adhesive Pest Traps**
     - **Definition**: Chemical trapping products with sticky surfaces, pest attraction, and physical entrapment capability.
     - **OUTPUT_LABEL**: 粘胶型虫鼠诱捕

### Personal Protection Products

151. **Mosquito Repellent Bracelet**
     - **Definition**: Wearable devices with repellent compounds providing personal mosquito protection through continuous wearing.
     - **OUTPUT_LABEL**: 驱蚊手环

152. **Mosquito Repellent Patches**
     - **Definition**: Adhesive patches with repellent agents providing localized insect protection through skin application.
     - **OUTPUT_LABEL**: 驱蚊贴

153. **Mosquito Repellent Liquid/Lotion**
     - **Definition**: Topical liquid/cream formulations with insect deterrent compounds for skin application and bite prevention.
     - **OUTPUT_LABEL**: 驱蚊液/乳液

### Cleaning and Disinfection

154. **Non-Antibacterial Daily Chemicals**
     - **Definition**: Household chemical products with cleaning capability but no antimicrobial claims for general household use.
     - **OUTPUT_LABEL**: 日化-非抑菌

155. **Daily Chemical Cleaners - Korean**
     - **Definition**: Household cleaning products with Korean origin, cleaning formulation, and household application purposes.
     - **OUTPUT_LABEL**: 日化清洁品（韩国）

156. **Antibacterial Daily Chemicals**
     - **Definition**: Household products with antimicrobial agents, disinfection capability, and bacterial elimination properties.
     - **OUTPUT_LABEL**: 日化-抑菌

157. **Other Water Purification Tablets**
     - **Definition**: Chemical tablets with water treatment compounds and dissolution capability for pools and aquariums.
     - **OUTPUT_LABEL**: 其他净水片

158. **Alcohol Cotton Pads/Alcohol Wipes**
     - **Definition**: Pre-moistened disposable materials with alcohol saturation for disinfection capability and cleaning.
     - **OUTPUT_LABEL**: 日用酒精棉片/酒精湿巾

159. **Disinfectants**
     - **Definition**: Chemical agents with antimicrobial activity designed for pathogen elimination and surface/fabric treatment.
     - **OUTPUT_LABEL**: 消毒剂

160. **Disinfecting Wipes**
     - **Definition**: Pre-moistened disposable wipes with disinfectant saturation for convenient cleaning and sanitization.
     - **OUTPUT_LABEL**: 消毒湿巾

161. **Pool Chlorine Tablets**
     - **Definition**: Chemical tablets with chlorine content and slow dissolution for swimming pool water sanitization.
     - **OUTPUT_LABEL**: 泳池氯片

### Agricultural Chemicals

162. **Chemical Fertilizers**
     - **Definition**: Synthetic nutrients including phosphorus, nitrogen, potassium, and compound fertilizers for plant growth enhancement.
     - **OUTPUT_LABEL**: 化肥

163. **Plant Growth Regulators**
     - **Definition**: Chemical compounds with hormonal activity for modifying plant growth and development processes.
     - **OUTPUT_LABEL**: 植物生长调节剂

## Safety Equipment and Protective Gear

### Hazardous Material Safety

164. **Radioactive Equipment**
     - **Definition**: Devices containing or emitting radioactive materials, UV, infrared, laser, microwave, or sonic waves.
     - **OUTPUT_LABEL**: 放射性设备

165. **Professional Climbing Carabiners**
     - **Definition**: Professional-grade carabiners with load-bearing design, climbing application, and professional certification for high-altitude activities.
     - **OUTPUT_LABEL**: 专业登山扣

166. **Handheld Brush Cutters & Accessories**
     - **Definition**: Handheld vegetation cutting tools with metal cutting blades and vegetation removal capability, prohibited in EU+UK.
     - **OUTPUT_LABEL**: 手握除草机&配件

## Clothing and Accessories

### Safety Hazard Clothing

167. **Adult Electric Clothing**
     - **Definition**: Adult clothing items with electronic components, heating/cooling functions, and special effects attachments.
     - **OUTPUT_LABEL**: 成人带电服装

168. **Iron Powder Heating Insoles**
     - **Definition**: Shoe insoles using iron powder oxidation for chemical heating and foot warming applications.
     - **OUTPUT_LABEL**: 铁粉加热鞋垫

169. **Clothing with Puncture/Scratch Hazards**
     - **Definition**: Clothing items with sharp decorative elements, protruding components, and injury potential from cutting or puncturing.
     - **OUTPUT_LABEL**: 有刺伤,划伤隐患的服装

170. **Other Safety Hazard Clothing Accessories**
     - **Definition**: Clothing accessories with various safety risks and potential harm to users through design hazards.
     - **OUTPUT_LABEL**: 其他安全隐患类服装配饰

### Children's Clothing

171. **Children's Shoes**
     - **Definition**: Footwear for children with foot length ≤240mm, featuring child-appropriate sizing and age-appropriate design.
     - **OUTPUT_LABEL**: 童鞋

172. **Children's Bicycle Seats**
     - **Definition**: Bicycle seats with child-appropriate sizing, safety restraints, and bicycle mounting systems for children.
     - **OUTPUT_LABEL**: 儿童自行车座椅

173. **Adult Electric Function Clothing**
     - **Definition**: Adult clothing with electrical heating/cooling functions including vests and specialized functional garments.
     - **OUTPUT_LABEL**: 成人电功能服

174. **Children's Electric Clothing**
     - **Definition**: Children's clothing with electrical components suitable for costume categories and dress-up purposes.
     - **OUTPUT_LABEL**: 儿童带电服装

175. **Children's Sleepwear**
     - **Definition**: Nightwear and home clothing with sleep-appropriate design, child sizing, and comfort materials.
     - **OUTPUT_LABEL**: 儿童睡衣

176. **Children's Costume Clothing**
     - **Definition**: Dress-up and costume clothing with character themes, play-oriented design, and child sizing.
     - **OUTPUT_LABEL**: 儿童造型服

177. **Corded Window Blinds - US Regulation**
     - **Definition**: Window treatments with cord mechanisms subject to US safety regulations due to strangulation hazard potential.
     - **OUTPUT_LABEL**: 美国带绳窗帘

178. **Infant Sleepwear**
     - **Definition**: Sleepwear specifically designed for infants and toddlers with sleep safety features and soft materials.
     - **OUTPUT_LABEL**: 婴儿睡衣

### Accessories

179. **Magnetic Nose/Tongue Studs**
     - **Definition**: Magnetic piercings with magnetic attachment and removable design that simulate piercing without actual piercing.
     - **OUTPUT_LABEL**: 磁吸鼻钉/舌钉

180. **Hydrocarbon-Containing Keychains**
     - **Definition**: Keychains with visible liquid contents containing flowing liquids that pose health risks if ingested.
     - **OUTPUT_LABEL**: 含烃类钥匙扣

## Children's Safety Products and Toys

### Sleep and Rest Safety

181. **Enclosed Baby Walkers**
     - **Definition**: Traditional baby walkers with surrounding frames, wheel mobility, and seated design for infant mobility.
     - **OUTPUT_LABEL**: 包围式学步车

182. **Nursing Pillow**
     - **Definition**: Specialized pillows with curved design for feeding support during breastfeeding for mothers and infants.
     - **OUTPUT_LABEL**: 哺乳枕

183. **Baby Bedding with Small Particles**
     - **Definition**: Bedding items with small decorative elements like beads or tassels posing choking hazard potential.
     - **OUTPUT_LABEL**: 带小颗粒的婴儿床品

184. **Inflatable Neck Float for Children**
     - **Definition**: Inflatable flotation devices worn around children's necks with buoyancy aid but strangulation risk potential.
     - **OUTPUT_LABEL**: 儿童充气颈部浮漂

185. **Non-Mesh Crib Bumpers**
     - **Definition**: Solid crib bumpers with solid construction and crib attachment but suffocation risk potential.
     - **OUTPUT_LABEL**: 非网状床围

186. **Infant Swaddling Sleep Products**
     - **Definition**: Swaddling products including baby nests, sleep pods, and bed-in-bed accessories with wrapping design.
     - **OUTPUT_LABEL**: 婴儿包裹类睡眠用品

187. **Infant Car Seats**
     - **Definition**: Specialized safety seats with safety harnesses and vehicle installation for infant protection during transportation.
     - **OUTPUT_LABEL**: 婴儿汽车座椅

188. **Infant Swaddles**
     - **Definition**: Wrapping cloths with wrapping design for restricting infant movement during sleep applications.
     - **OUTPUT_LABEL**: 婴儿襁褓

189. **Infant Inclined Sleep Products**
     - **Definition**: Sleep products with inclined surfaces positioning infants at angles during rest periods.
     - **OUTPUT_LABEL**: 婴儿倾斜睡眠产品

190. **Infant Sleep Sacks**
     - **Definition**: Wearable blankets with wearable design and sleep confinement for infant sleeping safety.
     - **OUTPUT_LABEL**: 婴儿睡袋

191. **Infant Sleep Companion Toys**
     - **Definition**: Toys with soft construction intended to accompany children during sleep but pose suffocation risk.
     - **OUTPUT_LABEL**: 婴儿睡眠陪伴玩偶

192. **Infant Sleep Head Pillows**
     - **Definition**: Pillows designed for infant head positioning and sleep support but pose suffocation risk potential.
     - **OUTPUT_LABEL**: 婴儿睡眠头枕

193. **Infant Blankets**
     - **Definition**: Blankets with infant-appropriate size for sleep application but pose suffocation risk concerns.
     - **OUTPUT_LABEL**: 婴儿毯

194. **Infant Mosquito Net Beds**
     - **Definition**: Beds with integrated mosquito netting, soft padding, and complete enclosed sleep environments.
     - **OUTPUT_LABEL**: 婴儿蚊帐床

195. **Infant Self-Feeding Products**
     - **Definition**: Products like self-feeding pillows with unattended feeding capability but pose choking hazard potential.
     - **OUTPUT_LABEL**: 婴儿自助喂食类商品

### Dangerous Toys

196. **Blowing Tools**
     - **Definition**: Toys or tools requiring forceful blowing action with respiratory challenge and potential choking hazard.
     - **OUTPUT_LABEL**: 吹气工具

197. **Magnetic Balls**
     - **Definition**: Small magnetic spheres with strong magnets and small size that pose high ingestion risk.
     - **OUTPUT_LABEL**: 磁力球01

198. **Realistic Food Toys**
     - **Definition**: Toy foods with food-like appearance and realistic detail that cause ingestion confusion potential.
     - **OUTPUT_LABEL**: 仿真食物

199. **Dart Fidget Spinner**
     - **Definition**: Fidget spinners with sharp dart-like projections, spinning mechanism, and injury potential from sharp elements.
     - **OUTPUT_LABEL**: 飞镖指尖陀螺

200. **Outdoor Toy Guns with Projectiles**
     - **Definition**: Toy firearms with projectile firing capability, realistic gun appearance, and eye/injury hazard potential.
     - **OUTPUT_LABEL**: 户外玩具枪（BB弹/橡胶弹）塑料射击玩具凝胶枪凝胶球爆弹弹药

201. **Flame Patterns**
     - **Definition**: Children's clothing featuring flame imagery or fire-related graphics inappropriate for children and safety concern symbolism.
     - **OUTPUT_LABEL**: 火焰图案

202. **Particle-Containing Teethers**
     - **Definition**: Teething toys with internal particles and loose elements posing choking hazard during teething application.
     - **OUTPUT_LABEL**: 颗粒牙胶

203. **Mushroom-Shaped Baby Teethers/Pacifiers**
     - **Definition**: Pacifiers or teethers with mushroom shapes designed for infant oral application but pose choking risks.
     - **OUTPUT_LABEL**: 蘑菇形婴儿磨牙器或安抚奶嘴

204. **Water-Absorbing Gel Beads/Capsules**
     - **Definition**: Small gel beads with water expansion capability and small initial size posing intestinal blockage potential.
     - **OUTPUT_LABEL**: 吸水凝珠（胶囊）

### General Toy Safety

205. **Magnetic Drawing Board**
     - **Definition**: Drawing toys with magnetic drawing surfaces, stylus tools, and erasable capability using magnetic mechanisms.
     - **OUTPUT_LABEL**: 磁性画板

206. **Magnetic Building Block Toys**
     - **Definition**: Construction toys with magnetic attachment and building capability but potential swallowing hazard from magnets.
     - **OUTPUT_LABEL**: 磁性积木玩具

207. **Magnetic Game Board**
     - **Definition**: Board games with magnetic game pieces, board format, and interactive play using magnetic mechanisms.
     - **OUTPUT_LABEL**: 磁性游戏板

208. **Children's Swimming Rings**
     - **Definition**: Inflatable flotation devices with inflatable design and buoyancy aid for children's swimming assistance.
     - **OUTPUT_LABEL**: 儿童游泳圈

209. **Building Block Toys**
     - **Definition**: Construction toys with fewer than 300 interlocking pieces classified as infant/toddler toys.
     - **OUTPUT_LABEL**: 积木玩具

210. **Stress Relief Soft Toys**
     - **Definition**: Squeezable toys with soft, squeezable material for stress relief purpose and tactile stimulation.
     - **OUTPUT_LABEL**: 减压软玩具

211. **Plush Dolls**
     - **Definition**: Soft toys with plush fabric exterior, stuffed construction, and character design for comfort.
     - **OUTPUT_LABEL**: 毛绒公仔

212. **Plush Hanging Accessories**
     - **Definition**: Small plush items with hanging attachments for decorative purpose and plush construction.
     - **OUTPUT_LABEL**: 毛绒挂件吊饰

213. **Water Guns**
     - **Definition**: Toy firearms with water propulsion mechanism and gun-like design for recreational water play.
     - **OUTPUT_LABEL**: 水枪

214. **Toy Foam Blaster Guns**
     - **Definition**: Toy guns with foam projectile firing, blaster mechanism, and recreational shooting capability.
     - **OUTPUT_LABEL**: 玩具泡沫爆破枪

215. **Paper Puzzles**
     - **Definition**: Jigsaw puzzles with fewer than 300 paper pieces and interlocking design for young children.
     - **OUTPUT_LABEL**: 纸质拼图

216. **Letter Puzzles**
     - **Definition**: Educational puzzles featuring alphabet letters with fewer than 300 pieces for letter recognition.
     - **OUTPUT_LABEL**: 字母拼图

217. **Infant and Toddler Toys**
     - **Definition**: All toys with age-appropriate design, safety considerations, and developmental benefits for infants and toddlers.
     - **OUTPUT_LABEL**: 婴童玩具

218. **Toy Balloons**
     - **Definition**: Inflatable rubber or latex balloons with inflatable design posing choking/suffocation hazard when deflated.
     - **OUTPUT_LABEL**: 玩具气球

## Baby and Infant Care Products

### Furniture and Equipment

219. **Rocking Horses**
     - **Definition**: Traditional riding toys with rocking motion, horse design, and balance requirement for children.
     - **OUTPUT_LABEL**: 棒马

220. **High Chairs and Cushions**
     - **Definition**: Elevated seating furniture with height elevation for infant feeding and sitting but fall hazard potential.
     - **OUTPUT_LABEL**: 高脚椅及坐垫

221. **Step Stools**
     - **Definition**: Small elevated platforms with step-up design for height assistance but tip-over potential.
     - **OUTPUT_LABEL**: 脚踏凳

222. **Safety Gates**
     - **Definition**: Barrier devices with doorway mounting and barrier function for restricting infant movement through doorways.
     - **OUTPUT_LABEL**: 门栏

223. **Baby Walkers**
     - **Definition**: Wheeled devices with wheel mobility for walking assistance but tip-over/stair hazard potential.
     - **OUTPUT_LABEL**: 学步车

224. **Rocking Horses and Animals**
     - **Definition**: Animal-shaped toys with animal design, rocking mechanism, and balance challenge for children.
     - **OUTPUT_LABEL**: 摇摆的马和动物

225. **Cradle Beds**
     - **Definition**: Small beds with rocking motion for infant sleep application but tip-over potential concern.
     - **OUTPUT_LABEL**: 摇篮床

226. **Rocking Chairs and Swings**
     - **Definition**: Seating with motion mechanism and seating design providing rocking or swinging motion for children.
     - **OUTPUT_LABEL**: 摇椅和秋千

227. **Baby Safety Harnesses and Leashes**
     - **Definition**: Restraint systems with harness design and movement restriction for controlling infant/toddler movement.
     - **OUTPUT_LABEL**: 婴儿安全带和牵引绳

228. **Baby Carriers and Accessories**
     - **Definition**: Wearable devices with body attachment for carrying infants against the body with hands-free mobility.
     - **OUTPUT_LABEL**: 婴儿背带和配件

229. **Baby Strollers**
     - **Definition**: Wheeled devices with wheel mobility and infant seating for transporting infants and toddlers.
     - **OUTPUT_LABEL**: 婴儿车

230. **Stroller Accessories**
     - **Definition**: Additional components with stroller compatibility and functional enhancement for baby stroller safety considerations.
     - **OUTPUT_LABEL**: 婴儿车配件

231. **Baby Cribs**
     - **Definition**: Enclosed sleeping furniture with enclosed design for infant sleep application requiring safety rail compliance.
     - **OUTPUT_LABEL**: 婴儿床

232. **Stroller Storage**
     - **Definition**: Storage compartments with storage capability and stroller integration providing convenience features for strollers.
     - **OUTPUT_LABEL**: 婴儿推车储存

233. **Baby Seat Accessories**
     - **Definition**: Additional components with seat compatibility for safety enhancement and comfort features in infant seating.
     - **OUTPUT_LABEL**: 婴儿座椅配件

234. **Play Yards**
     - **Definition**: Portable enclosed areas with enclosed play area and portability for infant play and containment.
     - **OUTPUT_LABEL**: 游戏床

235. **Toddler Beds**
     - **Definition**: Small beds with low height design and toddler sizing for transition from cribs.
     - **OUTPUT_LABEL**: 幼儿床

### Care Products

236. **Baby Tableware**
     - **Definition**: Feeding utensils and dishes with infant-appropriate sizing and safe materials for feeding assistance.
     - **OUTPUT_LABEL**: 婴儿餐具

237. **Baby Cotton Products**
     - **Definition**: Cotton-based products with cotton material and soft texture for infant care and hygiene applications.
     - **OUTPUT_LABEL**: 婴儿棉品

238. **Baby Skin Care**
     - **Definition**: Cosmetic and care products with gentle formulation and infant skin compatibility for protective purposes.
     - **OUTPUT_LABEL**: 婴儿皮肤护理

239. **Baby Toothbrushes**
     - **Definition**: Dental hygiene tools with soft bristles and infant-appropriate size for infant oral care.
     - **OUTPUT_LABEL**: 婴儿牙刷

240. **Baby Activity and Play Mats**
     - **Definition**: Padded surfaces with padded surface and developmental activities for infant play and motor development.
     - **OUTPUT_LABEL**: 婴儿运动和游戏垫

241. **Baby Products**
     - **Definition**: General category for all infant care and safety products with infant-specific design and care purposes.
     - **OUTPUT_LABEL**: 婴儿用品

## Consumer Goods and Materials

### Plastics and Disposables

242. **Sequins/Glitter/Microbeads**
     - **Definition**: Small plastic decorative particles with plastic composition and decorative purpose prohibited in some countries.
     - **OUTPUT_LABEL**: 亮片/亮粉/微珠

243. **Wet Wipes**
     - **Definition**: Pre-moistened disposable cleaning cloths with moisture saturation and disposable format for cleaning capability.
     - **OUTPUT_LABEL**: 湿巾

244. **Special Plastic Material Products**
     - **Definition**: Products made from plastics with visible black spots, defects, and quality concerns in construction.
     - **OUTPUT_LABEL**: 特殊塑料材质用品

245. **Disposable Laminated Paper Tableware**
     - **Definition**: Paper dishes and utensils with paper base, plastic coating, and single-use design.
     - **OUTPUT_LABEL**: 一次性淋膜类纸质餐具

246. **Disposable Food Packaging**
     - **Definition**: Single-use containers with food contact approval and disposable design for packaging purposes.
     - **OUTPUT_LABEL**: 一次性食品包装

247. **Disposable Plastic Items - Cotton Swabs/Balloon Sticks**
     - **Definition**: Single-use plastic items with plastic construction and single-use application having environmental impact.
     - **OUTPUT_LABEL**: 一次性塑料（棉签/气球棒）

248. **Disposable Plastic Tableware**
     - **Definition**: Single-use eating and serving utensils with plastic construction and eating utensil design.
     - **OUTPUT_LABEL**: 一次性塑料餐具

249. **Disposable Plastic Trays**
     - **Definition**: Single-use plastic serving and carrying trays with tray design and plastic material.
     - **OUTPUT_LABEL**: 一次性塑料架托

250. **Disposable Plastic Stirring Sticks**
     - **Definition**: Single-use plastic sticks with stirring design and plastic construction for beverage mixing.
     - **OUTPUT_LABEL**: 一次性塑料搅拌棒

251. **Disposable Plastic Food Containers**
     - **Definition**: Single-use plastic containers with food container design and plastic material for food storage.
     - **OUTPUT_LABEL**: 一次性塑料食品容器

### Home and Garden

252. **Portable Alcohol Fireplace**
     - **Definition**: Small, movable fireplaces with portable design using alcohol fuel and open flame operation.
     - **OUTPUT_LABEL**: 可移动酒精壁炉

253. **Decorative Balloons**
     - **Definition**: Balloons for decoration including custom, party, and wedding balloons with decorative purpose and event application.
     - **OUTPUT_LABEL**: 装饰性气球

254. **Faucets and Extension Accessories**
     - **Definition**: Water control devices with water flow control and plumbing integration with functional accessories.
     - **OUTPUT_LABEL**: 水龙头及延伸配件

### Food and Consumption

255. **Food & Beverages**
     - **Definition**: Shelf-stable foods and beverages with preserved composition and room temperature stability for consumption.
     - **OUTPUT_LABEL**: 食品&饮料

256. **Food & Health Supplements**
     - **Definition**: Health supplements with health benefit claims and regulatory approval for supplemental nutrition.
     - **OUTPUT_LABEL**: 食品＆保健品

257. **General Food Contact**
     - **Definition**: All food contact materials except disposable plastic tableware with food safety approval and contact compatibility.
     - **OUTPUT_LABEL**: 普通食品接触

### Publications

258. **Books**
     - **Definition**: Printed publications including adult books, magazines, children's literature, and comics with educational/entertainment purpose.
     - **OUTPUT_LABEL**: 图书类

## Adult Products and Content

259. **Insertable Catheters**
     - **Definition**: Adult insertable stimulation devices with tube-like design and specialized opening configuration for adult use.
     - **OUTPUT_LABEL**: 插入式导管

260. **Male Enhancement Products**
     - **Definition**: Products claiming male sexual performance enhancement including oils, gels, and supplements targeting adult males.
     - **OUTPUT_LABEL**: 壮阳增大产品

261. **Adult Products**
     - **Definition**: General category for adult-oriented intimate products with intimate purpose and age-restricted access.
     - **OUTPUT_LABEL**: 成人用品

## Restricted Content and Imagery

262. **Firearm/Weapon Related**
     - **Definition**: Content featuring firearms, weapon components, or 3D printing instructions with weapon imagery and manufacturing instructions.
     - **OUTPUT_LABEL**: 枪支武器相关

263. **Revealing/Exposing Products**
     - **Definition**: Products with mildly suggestive or revealing content featuring suggestive imagery and adult-oriented appeal.
     - **OUTPUT_LABEL**: 暴露款商品

264. **Highly Revealing Products**
     - **Definition**: Products with explicit or highly revealing content featuring explicit imagery and high exposure levels.
     - **OUTPUT_LABEL**: 推荐暴露款

265. **Disgusting/Offensive Products**
     - **Definition**: Products featuring disgusting or offensive imagery including fake insects, excrement toys, or medical imagery.
     - **OUTPUT_LABEL**: 推荐恶心商品

266. **Horror/Scary Products**
     - **Definition**: Products with frightening content including gore, horror masks, or violent imagery with disturbing content.
     - **OUTPUT_LABEL**: 推荐恐怖商品

## Beauty and Personal Care

267. **Pimple Patches**
     - **Definition**: Covering patches with adhesive application designed to conceal pimples and skin blemishes for cosmetic covering.
     - **OUTPUT_LABEL**: 痘痘贴

268. **Cosmetics**
     - **Definition**: Beauty and personal care products with cosmetic formulation for appearance enhancement and beauty applications.
     - **OUTPUT_LABEL**: 化妆品

##FINAL_REMINDER##
**Multi-label output is PREFERRED and EXPECTED from ##REFERENCE_CATEGORIES##.**
- Don't hesitate to include multiple relevant categories
- Comprehensive classification serves compliance better than restrictive single labels  
- **Think: "What are ALL the ways this item could be problematic based on ##REFERENCE_CATEGORIES##?"**
- **When in doubt about any category from ##REFERENCE_CATEGORIES##, include rather than exclude**
- Your comprehensive coverage helps human reviewers make informed decisions
    """
    
    # User prompt - use custom prompt if provided, otherwise use default
    user_prompt = """
    Here are examples of good images classifications:

Example 1: [Image of tactical folding knife with finger ring]
{"result":"刀具,爪刀"}

Example 2: [Image of children's cartoon backpack with size label]
{"result":"儿童包"}

Example 3: [Image of butterfly knife with dual handles]
{"result":"蝴蝶刀,刀具"}

Now analyze the image and classify the item(s) according to the categories defined in the system, following the same multi-label approach shown in the examples above.

**IMPORTANT**
- Output 5 labels AT MOST whenever multiple categories from ##REFERENCE_CATEGORIES##
- Output lables from the highest confidences to lower
    """
    
    try:
        # Determine if input is URL or local file
        if image_input.startswith(('http://', 'https://')):
            # Handle URL
            base64_media, content_type = encode_image_from_url(image_input)
            
            # Determine format from content-type or URL extension
            if 'png' in content_type.lower() or image_input.lower().endswith('.png'):
                file_extension = '.png'
            elif 'gif' in content_type.lower() or image_input.lower().endswith('.gif'):
                file_extension = '.gif'
            elif 'webp' in content_type.lower() or image_input.lower().endswith('.webp'):
                file_extension = '.webp'
            else:
                # Default to jpeg for most cases
                file_extension = '.jpeg'
            
            media_type = "image"
        else:
            # Handle local file - detect actual format from content
            try:
                actual_format, needs_conversion = detect_image_format(image_input)
                media_type = "image"
                
                if needs_conversion:
                    print(f"🔄 Converting {image_input} to JPEG format...")
                    base64_media = encode_image_to_base64(image_input, convert_to_jpeg=True)
                else:
                    base64_media = encode_image_to_base64(image_input, convert_to_jpeg=False)
                
                file_extension = f'.{actual_format}'
            except Exception as e:
                raise Exception(f"Failed to process image file: {str(e)}")
        
        # Create Bedrock client with credential handling
        # Priority: function parameters > global variables > environment variables > AWS credentials file
        access_key = aws_access_key_id or AWS_ACCESS_KEY_ID or os.getenv('AWS_ACCESS_KEY_ID')
        secret_key = aws_secret_access_key or AWS_SECRET_ACCESS_KEY or os.getenv('AWS_SECRET_ACCESS_KEY')
        
        if access_key and secret_key:
            client = boto3.client(
                "bedrock-runtime", 
                region_name=region,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key
            )
            print(f"Using explicit AWS credentials for region: {region}")
        else:
            # Use default credential chain (environment variables, AWS credentials file, IAM roles, etc.)
            client = boto3.client("bedrock-runtime", region_name=region)
            print(f"Using default AWS credential chain for region: {region}")
        
        # Message content for converse API
        # For converse API, we need to decode base64 back to bytes
        import base64 as b64
        image_bytes = b64.b64decode(base64_media)
        
        # Convert file extension to AWS Bedrock format
        format_mapping = {
            '.jpg': 'jpeg',
            '.jpeg': 'jpeg', 
            '.png': 'png',
            '.gif': 'gif',
            '.webp': 'webp'
        }
        bedrock_format = format_mapping.get(file_extension, 'jpeg')  # Default to jpeg for unknown formats
        
        message_content = [
            {media_type: {"format": bedrock_format, "source": {"bytes": image_bytes}}},
            {"text": user_prompt}
        ]
        
        # Configure system prompt with optional caching
        system_config = [{'text': system_prompt}]
        if use_cache:
            system_config.append({'cachePoint': {'type': 'default'}})
        
        # Retry mechanism with exponential backoff
        max_retries = 3
        base_delay = 1
        
        for attempt in range(max_retries + 1):
            try:
                # Use converse API with optional prompt caching
                response = client.converse(
                    modelId=model_id,
                    messages=[
                        {
                            'role': 'user',
                            'content': message_content
                        },
                        {
                            'role': 'assistant',
                            'content': [{'text': 'Here are the classification result:\n```json'}]
                        }
                    ],
                    system=system_config,
                    inferenceConfig={
                        'maxTokens': 150,
                        'topP': 0.01,
                        'temperature': 0
                    }
                )
                break  # Success, exit retry loop
                
            except Exception as e:
                error_str = str(e)
                if attempt < max_retries and ('ThrottlingException' in error_str or 'Too many tokens' in error_str or 'ServiceUnavailable' in error_str):
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    print(f"⏳ Retry {attempt + 1}/{max_retries} after {delay}s due to: {error_str.split(':')[-1].strip()}")
                    time.sleep(delay)
                else:
                    raise e  # Re-raise if max retries reached or non-retryable error
        
        # Extract token usage metrics
        usage = response['usage']
        input_tokens = usage.get('inputTokens', 0)
        output_tokens = usage.get('outputTokens', 0)
        total_tokens = input_tokens + output_tokens
        
        # Create metrics dictionary with cache info
        metrics = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens
        }
        
        # Add cache metrics if available
        if 'cacheCreationInputTokens' in usage:
            metrics['cache_creation_tokens'] = usage['cacheCreationInputTokens']
        if 'cacheReadInputTokens' in usage:
            metrics['cache_read_tokens'] = usage['cacheReadInputTokens']
        
        # Print token metrics with cache info
        cache_info = ""
        if 'cache_creation_tokens' in metrics:
            cache_info += f", Cache Created: {metrics['cache_creation_tokens']}"
        if 'cache_read_tokens' in metrics:
            cache_info += f", Cache Hit: {metrics['cache_read_tokens']}"
        
        print(f"📊 Token Metrics - Input: {input_tokens}, Output: {output_tokens}, Total: {total_tokens}{cache_info}")
        
        # Extract generated text
        generated_text = response['output']['message']['content'][0]['text']
        
        # Return based on return_metrics flag
        if return_metrics:
            return generated_text, metrics
        else:
            return generated_text
        
    except Exception as e:
        raise Exception(f"Error in img_tagging: {str(e)}")

def analyze_image_simple(media_path, region="us-west-2", model_id="us.amazon.nova-lite-v1:0", 
                        aws_access_key_id=None, aws_secret_access_key=None, use_cache=True):
    """
    Backward compatibility wrapper - prints results like the original function
    """
    try:
        result = img_tagging(media_path, None, region, model_id, aws_access_key_id, aws_secret_access_key, use_cache=use_cache)
        
        # Print results in original format
        print("=== Generated Text Only ===")
        print(result)
        
    except Exception as e:
        print(f"Error: {str(e)}")

def process_excel_data(excel_file='resources/sampled_1000.xlsx', output_file='result.xlsx', 
                      images_dir='/Users/zeyao/Documents/Images/small', prompt=None, 
                      region="us-west-2", model_id="us.amazon.nova-lite-v1:0",
                      aws_access_key_id=None, aws_secret_access_key=None, use_cache=True):
    """
    Process Excel data with local image files and perform image tagging
    
    Args:
        excel_file: Input Excel file path
        output_file: Output Excel file path
        images_dir: Directory containing the local images
        prompt: Custom prompt for image analysis
        region: AWS region
        model_id: Nova model ID
        aws_access_key_id: AWS access key (optional)
        aws_secret_access_key: AWS secret key (optional)
        use_cache: If True, enables prompt caching for system prompt (default: True)
    """
    # Read Excel file
    df = pd.read_excel(excel_file)
    
    # Create results list
    results = []
    
    # Token tracking variables
    total_input_tokens = 0
    total_output_tokens = 0
    successful_requests = 0
    failed_requests = 0
    content_filtered_requests = 0
    html_files = 0
    unsupported_formats = 0
    
    print(f"开始处理 {len(df)} 条数据...")
    print("=" * 60)
    
    # Process each row
    for index, row in df.iterrows():
        # Check if 'image' column exists, otherwise use first two columns as before
        if 'image' in df.columns:
            tag_gt = row.get('tag_gt', row.iloc[0])  # Try to get tag_gt column, fallback to first column
            image_filename = row['image']  # Get image filename from 'image' column
            image_path = os.path.join(images_dir, image_filename)
        else:
            # Fallback to original behavior for backward compatibility
            tag_gt = row.iloc[0]  # First column: tag_gt
            image_path = row.iloc[1]  # Second column: assume it's already a path
        
        try:
            # Check if local image file exists
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Call inference function with metrics using local image path
            result, metrics = img_tagging(image_path, prompt, region, model_id, aws_access_key_id, aws_secret_access_key, return_metrics=True, use_cache=use_cache)
            
            # Update token counters
            total_input_tokens += metrics["input_tokens"]
            total_output_tokens += metrics["output_tokens"]
            
            print(f"✅ 处理第{index+1}行: {image_path} -> {result}")
            successful_requests += 1
        except Exception as e:
            error_msg = str(e)
            result = f"错误: {error_msg}"
            
            # Categorize different types of errors
            if "HTML" in error_msg or "appears to be HTML" in error_msg:
                html_files += 1
                print(f"🌐 处理第{index+1}行 (HTML文件): {image_path}")
            elif "AVIF" in error_msg or "not supported" in error_msg:
                unsupported_formats += 1
                print(f"❓ 处理第{index+1}行 (不支持格式): {image_path}")
            else:
                print(f"❌ 处理第{index+1}行出错: {image_path} -> {result}")
            
            failed_requests += 1
        
        # Clean up result - remove markdown formatting and extract JSON
        result = result.replace("```", "").replace("json", "").strip()
        inference_result = result
        
        # Try to parse JSON result
        try:
            # Check for content filter responses
            if "content filters" in result.lower() or "blocked" in result.lower():
                inference_result = "CONTENT_FILTERED"
                print(f"⚠️  Content filtered response detected")
            else:
                # Find JSON part if there's extra text
                json_start = result.find('{')
                json_end = result.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    json_part = result[json_start:json_end]
                    obj = json.loads(json_part)
                    inference_result = obj["result"]
                else:
                    # Fallback: try parsing the whole result
                    obj = json.loads(result)
                    inference_result = obj["result"]
        except Exception as e:
            print(f"⚠️  failed to parse result: {str(e)[:50]}...")
            # Check if it's a content filter message
            if "content filters" in result.lower() or "blocked" in result.lower():
                inference_result = "CONTENT_FILTERED"
            else:
                # Keep the original result if JSON parsing fails
                inference_result = result
        
        # Track content filtered responses
        if inference_result == "CONTENT_FILTERED":
            content_filtered_requests += 1
        
        # Add to results list
        results.append({
            'tag_gt': tag_gt,
            'image_path': image_path,
            'inference_result': inference_result
        })
    
    # Create result DataFrame and save to Excel
    result_df = pd.DataFrame(results)
    result_df.to_excel(output_file, index=False)
    
    # Calculate average tokens per request
    avg_input_tokens = total_input_tokens / successful_requests if successful_requests > 0 else 0
    avg_output_tokens = total_output_tokens / successful_requests if successful_requests > 0 else 0
    total_tokens = total_input_tokens + total_output_tokens
    
    # Print final summary
    print("=" * 60)
    print(f"📋 处理完成总结:")
    print(f"   • 总处理数据: {len(results)} 条")
    print(f"   • 成功请求: {successful_requests} 条")
    print(f"   • 失败请求: {failed_requests} 条")
    print(f"     - HTML文件: {html_files} 条")
    print(f"     - 不支持格式: {unsupported_formats} 条")
    print(f"     - 其他错误: {failed_requests - html_files - unsupported_formats} 条")
    print(f"   • 内容过滤: {content_filtered_requests} 条")
    print(f"   • 结果已保存到: {output_file}")
    print(f"")
    print(f"🔢 Token 使用统计:")
    print(f"   • 总输入 Token: {total_input_tokens:,}")
    print(f"   • 总输出 Token: {total_output_tokens:,}")
    print(f"   • 总计 Token: {total_tokens:,}")
    print(f"   • 平均输入 Token/请求: {avg_input_tokens:.1f}")
    print(f"   • 平均输出 Token/请求: {avg_output_tokens:.1f}")
    print("=" * 60)

if __name__ == "__main__":
    # Example 1: Single image analysis without caching
    # print("\n=== Single Image Analysis (Cache OFF) ===")
    # analyze_image_simple(media_file, use_cache=False)
    
    # Example 2: Excel batch processing (uncomment to use)
    print("\n=== Excel Batch Processing ===")
    process_excel_data('resources/sampled_1000.xlsx', 'results/sampled_1000_result_v11_small.xlsx', use_cache=True)
    
    # Example 3: Excel processing with custom credentials (uncomment to use)
    # process_excel_data(
    #     excel_file='black_url_flag.xlsx',
    #     output_file='result.xlsx',
    #     aws_access_key_id="YOUR_ACCESS_KEY_HERE",
    #     aws_secret_access_key="YOUR_SECRET_KEY_HERE",
    #     use_cache=True
    # )
