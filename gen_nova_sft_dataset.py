

import pandas as pd
import json

# Constants
account_id = "687752207838"
s3_bucket = "687752207838-dify-files"
s3_prefix = "shein_img_tagging/imgs"
system_prompt = """# Image Classification Assistant
    
    ## Role
    You are an image analyzer specialized in classifying visual content according to provided reference categories.
    
    ## Analysis Framework
    You must follow this 4-step process for every image:
    
    ### Step 1: Physical Attribute Extraction
    - Extract the objective physical attributes of all items in the image.
    - Focus strictly on neutral, geometric, and material properties.
    - **DO NOT** interpret the item's purpose, function, or potential use.
    - **Examples of attributes to extract:**
    -   **Shape:** Cylindrical, rectangular, pointed tip, curved edge, ring-shaped.
    -   **Components:** Handle, blade, trigger, nozzle, switch, USB port, chain.
    -   **Material/Texture:** Metallic, plastic, fabric, wooden, smooth, serrated.
    -   **Text/Markings:** Any visible letters, numbers, or symbols.
    
    ### Step 2: Feature Analysis
    Based on your description, identify key features:
    - Look for specific shapes (cylindrical, pointed, etc.)
    - Identify materials (metal, plastic, fabric, etc.)
    - Note functional elements (buttons, switches, blades, etc.)
    - Check for size indicators or child-oriented designs
    
    ### Step 3: Category Matching
    Compare identified features with the reference categories:
    - Match observed features with the "Key Features" listed for each category
    - Consider the "Definition" to understand the intended use or design purpose
    - Determine if the item meets the criteria for any category
    
    ### Step 4: Final Classification
    Provide classification result following these rules:
    - Return only the Chinese category names (the values in parentheses)
    - If multiple categories apply, separate them with English commas ","
    - Maximum of 5 categories
    - If no categories match, return "无"
    - Format your response as JSON: {"result":"target"}
    
    ## Output Requirements
    - Perform Steps 1-3 internally (do not show your analysis)
    - Only output the final JSON result from Step 4
    - Use Chinese category names only
    - Follow the exact JSON format specified
    
    ## Important Guidelines
    - Be precise - only classify items that clearly match the definitions
    - Consider both definition and key features when classifying
    - If uncertain about a match, err on the side of caution and exclude it
    - Do NOT include escaped Unicode characters in your output

    ## Reference Categories

    ### Knives and Blades
    1. **匕首 (Dagger)**  
       - **Definition**: A short-bladed weapon designed primarily for thrusting or stabbing attacks, characterized by a sharp pointed tip and compact design optimized for single-hand use.
       - **Key Features**: Sharp pointed tip (essential), short blade, designed for thrusting, single or double-edged, compact for close-quarters use.
    
    2. **刀具 (Knives)**  
       - **Definition**: Includes all types of knives, such as utility, tactical, folding, or ornamental blades, designed for cutting or slashing.  
       - **Key Features**: Sharp blade edges, handle designs (with or without finger rings), sheath or folding mechanisms, blade-like structures.

    3. **弹簧刀 (Switchblade)**  
       - **Definition**: A folding knife with a blade that deploys automatically via a spring mechanism, often triggered by a button.  
       - **Key Features**: Button or lever on handle, blade concealed in handle when folded, rapid deployment action implied by design.
    
    4. **蝴蝶刀 (Butterfly Knife)**  
       - **Definition**: A folding pocket knife with two handles that counter-rotate around the tang, allowing the blade to be concealed or deployed.  
       - **Key Features**: No curved handle, dual pivoting handles, blade sandwiched between handles when closed, often metallic with latch mechanism.
    
    5. **伪装刀具 (Disguised Knife)**  
       - **Definition**: Knives concealed within everyday objects, such as pens, credit cards, or belts, to evade detection.  
       - **Key Features**: Hidden blade mechanism, innocuous outer appearance (e.g., lipstick case, keychain), deployable sharp edge.
    
    6. **爪刀 (Claw Knife)**  
       - **Definition**: A knife designed for slashing or gripping, featuring a curved or slightly straight blade with one or more finger rings or holes in the handle for enhanced grip and hand-worn capability. Includes tactical, folding, and ornamental variations, distinguished by the presence of finger rings.  
       - **Key Features**: One or more finger holes or rings in the handle as the primary identifier, curved or slightly straight blade.
    
    ### Weapons and Restricted Items
    
    7. **仿真子弹 (Imitation Bullet)**  
       - **Definition**: Replica ammunition, often used in keychains or necklaces (e.g., bullet keychains, bullet necklaces).  
       - **Key Features**: Cylindrical casing with pointed tip, metallic sheen, no live components.
    
    8. **警用喷雾 (Police Spray)**  
       - **Definition**: Aerosol sprays including pepper, chili, tear gas, or anti-wolf/safety sprays.  
       - **Key Features**: Canister with nozzle, warning labels, handheld design.
    
    9. **警棍/甩棍 (Baton/Swing Stick)**  
       - **Definition**: Prohibited metal batons made of high-strength alloy steel, often with aggressive protrusions, some with police markings.  
       - **Key Features**: Telescoping sections, metallic finish, protrusions (e.g., spikes, barbs), police insignia.
    
    10. **电击器/电击棒 (Stun Gun/Stun Baton)**  
        - **Definition**: Devices with exposed high-voltage electrodes and aggressive protrusions, producing electric arcs.  
        - **Key Features**: Exposed electrodes, protrusions (e.g., spikes, barbs), visible electric arc or light.
    
    11. **指虎铁莲花 (Knuckle Duster/Iron Lotus)**  
        - **Definition**: Hand-worn offensive weapons including knuckle dusters, push swords, wolverine claws, iron lotus, finger rings, or key sticks.  
        - **Key Features**: Metal rings for fingers, protruding spikes or blades, fist-enclosing design.
    
    ### Explosive or Hazardous Items
    
    12. **卡式炉 (Cassette Stove)**  
        - **Definition**: A portable cooking stove using gas canisters.  
        - **Key Features**: Burner head, gas canister slot, foldable legs, ignition switch.
    
    13. **烟花爆竹 (Fireworks and Firecrackers)**  
        - **Definition**: A pyrotechnic product that produces bright sparks by igniting a chemical coating.
        - **Key Features**: Rod/cylinder; produces sparks and flame; requires ignition; often has a pointed base for insertion; typically covered in glitter/metallic finish.
    
    14. **射钉弹 (Nail Gun Cartridge)**  
        - **Definition**: Ammunition for nail guns, containing explosive charges to drive nails.  
        - **Key Features**: Cylindrical cartridge with metallic tip, labeled as ammunition, compatible with nail gun devices.
    
    15. **打火石、打火棒 (Flint or Fire Starter)**  
        - **Definition**: Tools using flint or metal rods to generate sparks for fire starting.  
        - **Key Features**: Metallic rod or stone, striker tool, spark-generating surface.
    
    16. **气罐 (Gas Canister)**  
        - **Definition**: Pressurized containers banned for cross-border transport, containing flammable or compressed gases.  
        - **Key Features**: Cylindrical metal container, pressure valve, hazard labels, sealed cap.
    
    17. **压罐喷雾 (Aerosol Spray Can)**  
        - **Definition**: Pressurized cans dispensing flammable or hazardous sprays.  
        - **Key Features**: Metal canister, nozzle with trigger, pressure release valve, warning labels.
    
    18. **烟雾弹 (Smoke Bomb)**  
        - **Definition**: Devices releasing colored smoke, often used for signaling or entertainment.  
        - **Key Features**: Small cylindrical or spherical shape, fuse or ignition point, smoke-emitting vent.
    
    19. **固体酒精 (Solid Alcohol)**  
        - **Definition**: Solid fuel blocks used for heating or cooking.  
        - **Key Features**: Rectangular or disc-shaped blocks, wax or gel-like texture, flammable packaging.
    
    20. **镁粉 (Magnesium Powder)**  
        - **Definition**: Fine magnesium powder used as a flammable material.  
        - **Key Features**: Silvery powder, airtight packaging, flammable warning labels.
    
    21. **烟饼（片） (Smoke Cake/Piece)**  
        - **Definition**: Compact smoke-producing devices, often for rituals or effects.  
        - **Key Features**: Flat or disc shape, ignition point, smoke-emitting surface.
    
    22. **重燃蜡烛 (Relight Candles)**  
        - **Definition**: Candles designed to relight after being blown out.  
        - **Key Features**: Waxed wick with embedded relighting mechanism, candle shape, flammable material.
    
    23. **火柴 (Matches)**  
        - **Definition**: Small sticks with a combustible head for ignition.  
        - **Key Features**: Wooden or paper sticks, colored ignitable tip, striking surface.
    
    24. **炸药 (Explosives)**  
        - **Definition**: High-explosive materials used for demolition or blasting.  
        - **Key Features**: Block or granular form, industrial packaging, explosive hazard symbols, detonation wiring.
    
    25. **红磷、白磷 (Red Phosphorus/White Phosphorus)**  
        - **Definition**: Highly reactive phosphorus compounds used in explosives or incendiaries.  
        - **Key Features**: Red or white crystalline powder, sealed containers, chemical hazard labels.
    
    26. **火药 (Gunpowder)**  
        - **Definition**: A combustible powder used as a propellant or explosive.  
        - **Key Features**: Granular or fine powder appearance, packaging with explosive warnings, metallic or plastic containers.
    
    27. **礼花筒 (Firework Tube)**  
        - **Definition**: A celebratory device that ejects confetti or streamers through mechanical action (non-combustible).
        - **Key Features**: Cylindrical tube; ejects paper/streamers; no fire or heat; often has twisting/pushing mechanism at base.
    
    ### Electronics and Appliances
    
    28. **充电宝 (Power Bank)**  
        - **Definition**: A portable device used to charge other electronic devices via a built-in battery.  
        - **Key Features**: Rectangular or cylindrical shape, USB ports (input/output), battery indicator lights, compact design.
    
    29. **纯锂电池 (Pure Lithium Battery)**  
        - **Definition**: Standalone lithium batteries, identifiable by CR/BR markings or labeled as "lithium battery/锂电池," excluding those sold with other products.  
        - **Key Features**: CR/BR prefix on labeling, cylindrical or coin-shaped, metallic casing, no accompanying devices.
    
    30. **电子打火器 (Electronic Lighter)**  
        - **Definition**: Lighters using piezoelectric crystals to generate high-voltage sparks, including rechargeable or battery-powered models.  
        - **Key Features**: Pressable crystal or button, electrode gap for sparks, charging port or battery compartment.
    
    31. **打火器（无燃气） (Gas-Free Lighter)**  
        - **Definition**: Lighters or electronic ignition devices without gas fuel, including flint or piezoelectric types.  
        - **Key Features**: Flint wheel or piezoelectric crystal, no visible gas tank, compact ignition design.
    
    32. **燃气打火机 (Gas Lighter)**  
        - **Definition**: Lighters powered by flammable gas (e.g., butane).  
        - **Key Features**: Gas tank visible, ignition trigger, flame adjustment knob, compact design.

    ### Children's Products
    33. **儿童包 (Children's Bag)**  
        - **Definition**: A bag designed specifically for children, varying in size and purpose, used for carrying personal items, school supplies, or toys.  
        - **Key Features**: Playful designs, cartoon characters, or themed motifs. Secure zipper closure. Various carrying options (wrist, handle, shoulder). Child-sized, with compartments or single storage, sometimes with charms.

    34. **童鞋 (Children's Shoes)**  
        - **Definition**: Shoes for children, indicated by text, images, or sizes (foot length ≤240mm).  
        - **Key Features**: Small size, child-friendly design, labeled size range.

    35. **婴童帽子 (Infant/Toddler Hat)**  
        - **Definition**: Headwear specifically designed for infants and toddlers, primarily for warmth or sun protection.  
        - **Key Features**: Soft, comfortable material, providing warmth or sun protection. Often features cartoon patterns, animal shapes, or cute prints. May include ear flaps, chin straps, or pom-pom decorations for comfort and secure fit.

    36. **儿童发饰 (Children's Hair Accessories)**  
        - **Definition**: Decorative hair clips, headbands, or hair ties specifically designed for children, used to secure or adorn hairstyles.  
        - **Key Features**: Brightly colored and varied in shape (e.g., bows, stars, butterflies), often featuring cartoon patterns or cute embellishments. Typically secured with clips, elastic bands, or ties, ensuring they are safe and easy to wear.

    37. **儿童太阳镜 (Children's Sunglasses)**  
        - **Definition**: Eyewear specifically designed for children to protect their eyes from the sun's harmful UV rays, often featuring playful designs.  
        - **Key Features**: UV protection, durable and child-friendly materials, various fun shapes (e.g., heart, flower, classic, cat-eye), colorful frames and lenses, and sometimes decorative elements like glitter or animal ears.

    38. **儿童自行车座椅 (Child Bicycle Seat)**  
        - **Definition**: A bicycle seat designed for children, indicated by text or images claiming child suitability.  
        - **Key Features**: Small seat attached to bike frame, child-sized design, safety straps or padding.

    ### Other Restricted Items
    39. **手铐、拇指烤、脚镣 (Handcuffs/Thumb Cuffs/Leg Irons)**  
        - **Definition**: Metal restraints meeting size criteria: handcuffs (>5cm inner diameter), thumb cuffs (>1cm), leg irons (>8cm).  
        - **Key Features**: Metal construction, hinged or linked design, size consistent with restraint purpose."""

# system_prompt = "请分析图片。"
user_prompt = "Please classify the item in this image according to the categories defined in the system."

def create_record(flag, filename):
    s3_path = f"s3://{s3_bucket}/{s3_prefix}/{filename}"

    img_format = "jpeg" if filename.endswith(".jpg") else filename.split('.')[-1]

    if img_format not in ['jpeg']:
        print(img_format)
        return None
    
    return {
        "schemaVersion": "bedrock-conversation-2024",
        "system": [{
            "text": system_prompt
        }],
        "messages": [{
                "role": "user",
                "content": [
                    {
                        "image": {
                            "format": img_format,
                            "source": {
                                "s3Location": {
                                    "uri": s3_path,
                                    "bucketOwner": account_id
                                }
                            }
                        }
                    },
                    {	"text": user_prompt	}
                ]
            },
            {
                "role": "assistant",
                "content": [{
                    "text": flag
                }]
            }
        ]
    }

# Load data
df = pd.read_excel('test_data.xlsx')

# Generate JSONL
with open('nova_sft_testset.jsonl', 'w', encoding='utf-8') as f:
    for idx, row in df.iterrows():
        gt_label = '{"result":"' + row['flag'] + '"}'

        record = create_record(gt_label, row['filename'])
        if record:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

print(f"Generated {len(df)} records in nova_sft_dataset.jsonl")