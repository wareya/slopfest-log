#!/usr/bin/env python3
import sys
import struct
import colorsys
import argparse

try:
    from PIL import Image
except ImportError:
    print("Error: The 'Pillow' library is required to create the PNG.")
    print("Install it by running: pip install Pillow")
    sys.exit(1)

def print_legend():
    """Prints a legend explaining the color generation to the console."""
    print("=== Color Legend ===")
    print("Mode 0 : Red          (16 Partitions)")
    print("Mode 1 : Orange       (64 Partitions)")
    print("Mode 2 : Yellow-Green (64 Partitions)")
    print("Mode 3 : Green        (64 Partitions)")
    print("Mode 4 : Cyan         (No Partitions - Solid)")
    print("Mode 5 : Blue         (No Partitions - Solid)")
    print("Mode 6 : Purple       (No Partitions - Solid)")
    print("Mode 7 : Magenta      (64 Partitions)")
    print("Invalid: Black")
    print("====================")

def main():
    parser = argparse.ArgumentParser(description="Visualize BC7 block modes and partitions in a DDS file.")
    parser.add_argument("input_dds", help="Path to the input BC7 DDS file.")
    parser.add_argument("output_png", help="Path to the output PNG file.")
    args = parser.parse_args()

    with open(args.input_dds, "rb") as f:
        # Read the standard 128-byte DDS header
        header = f.read(128)
        if len(header) < 128 or header[:4] != b"DDS ":
            print("Error: Not a valid DDS file.")
            sys.exit(1)

        # Offsets for standard DDS header
        height = struct.unpack("<I", header[12:16])[0]
        width = struct.unpack("<I", header[16:20])[0]
        fourcc = header[84:88]

        is_bc7 = False
        data_offset = 128

        # Check for DX10 Extension Header
        if fourcc == b"DX10":
            dx10_header = f.read(20)
            if len(dx10_header) < 20:
                print("Error: Invalid DX10 header.")
                sys.exit(1)
            
            dxgi_format = struct.unpack("<I", dx10_header[0:4])[0]
            # 97 = BC7_TYPELESS, 98 = BC7_UNORM, 99 = BC7_UNORM_SRGB
            if dxgi_format in (97, 98, 99):
                is_bc7 = True
            data_offset = 148
        # Some custom tools skip DX10 and use these FourCC codes instead
        elif fourcc in (b"BC7U", b"BC7L", b"BC7\x00"):
            is_bc7 = True

        if not is_bc7:
            print("Error: The provided DDS file is not BC7 encoded.")
            sys.exit(1)

        print(f"Detected BC7 Image: {width}x{height}")
        
        # BC7 blocks are 4x4 pixels
        bw = (width + 3) // 4
        bh = (height + 3) // 4

        f.seek(data_offset)

        # Create a 1 pixel = 1 block image first
        img = Image.new("RGB", (bw, bh))
        pixels = img.load()

        for y in range(bh):
            for x in range(bw):
                block = f.read(16) # Each BC7 block is 16 bytes (128 bits)
                if len(block) < 16:
                    break # Reached end of file or truncated data

                # Mode and Partition data fit entirely in the first two bytes
                val = struct.unpack("<H", block[:2])[0]
                
                # Determine mode (unary coded in the LSBs)
                mode = -1
                for i in range(8):
                    if (val & (1 << i)):
                        mode = i
                        break

                # Extract partition shape ID based on mode
                partition = -1
                if mode == 0:
                    partition = (val >> 1) & 0x0F # 4 bits
                elif mode == 1:
                    partition = (val >> 2) & 0x3F # 6 bits
                elif mode == 2:
                    partition = (val >> 3) & 0x3F # 6 bits
                elif mode == 3:
                    partition = (val >> 4) & 0x3F # 6 bits
                elif mode == 7:
                    partition = (val >> 8) & 0x3F # 6 bits

                # Calculate Color
                if mode == -1:
                    r, g, b = 0, 0, 0 # Invalid block
                else:
                    # Assign a distinct hue for each mode (0 to 7)
                    hue = mode / 8.0 
                    
                    if partition != -1:
                        # Modulate saturation and brightness to differentiate up to 64 partitions
                        s_step = partition % 8
                        v_step = partition // 8
                        sat = 0.3 + 0.7 * (s_step / 7.0) # Range 0.3 - 1.0
                        val_c = 0.3 + 0.7 * (v_step / 7.0) # Range 0.3 - 1.0
                    else:
                        # Modes 4, 5, and 6 have no partitions
                        sat = 1.0
                        val_c = 1.0

                    r, g, b = colorsys.hsv_to_rgb(hue, sat, val_c)
                    r, g, b = int(r * 255), int(g * 255), int(b * 255)

                pixels[x, y] = (r, g, b)

        # To make it readable, we scale it back up by 4x so 1 block = 4x4 pixels 
        # Using NEAREST filtering ensures block colors remain sharp and unblended
        img_scaled = img.resize((bw * 4, bh * 4), Image.NEAREST)
        
        # Crop exactly to the original image bounds (in case width/height weren't multiples of 4)
        img_cropped = img_scaled.crop((0, 0, width, height))
        
        img_cropped.save(args.output_png)
        print_legend()
        print(f"\nSuccessfully generated block visualization: {args.output_png}")

if __name__ == "__main__":
    main()
