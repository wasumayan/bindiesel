#!/usr/bin/env python3
"""
Generate AprilTag images
Creates printable AprilTag images for home marker detection
"""

import sys
import argparse
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError as e:
    print(f"ERROR: Missing required module: {e}")
    print("Install with: pip install opencv-python")
    sys.exit(1)

# Try to import AprilTag generation library
try:
    from apriltag import apriltag
    HAS_APRILTAG_GEN = True
except ImportError:
    try:
        import apriltag
        HAS_APRILTAG_GEN = True
    except ImportError:
        HAS_APRILTAG_GEN = False
        print("WARNING: AprilTag generation library not found.")
        print("Will use OpenCV's ArUco markers instead (similar functionality)")
        print("Install with: pip install dt-apriltags (for AprilTag) or use ArUco")


def generate_apriltag_image(tag_id=0, tag_family='tag36h11', size_px=200, border_bits=1, output_path=None, use_aruco=True):
    """
    Generate an ArUco marker image (or AprilTag if available)
    
    Args:
        tag_id: Tag ID to generate (0-249 for ArUco DICT_6X6_250)
        tag_family: Tag family (default: tag36h11) - ignored if using ArUco
        size_px: Output image size in pixels (default: 200)
        border_bits: Border width in bits (default: 1) - ignored if using ArUco
        output_path: Output file path (default: apriltag_{tag_id}.png)
        use_aruco: Use OpenCV ArUco instead of AprilTag (default: True)
    
    Returns:
        Generated image as numpy array
    """
    if use_aruco or not HAS_APRILTAG_GEN:
        # Use OpenCV ArUco markers (similar to AprilTag)
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        
        # Generate marker
        marker_image = np.zeros((size_px, size_px), dtype=np.uint8)
        marker_image = cv2.aruco.generateImageMarker(aruco_dict, tag_id, size_px, marker_image, 1)
        
        # Convert to 3-channel for saving
        marker_image_bgr = cv2.cvtColor(marker_image, cv2.COLOR_GRAY2BGR)
        
        if output_path:
            cv2.imwrite(str(output_path), marker_image_bgr)
            print(f"Generated ArUco marker image: {output_path}")
            print(f"  Marker ID: {tag_id}")
            print(f"  Dictionary: DICT_6X6_250")
            print(f"  Size: {size_px}x{size_px} pixels")
            print(f"  NOTE: Use ArUco detection in test_apriltag_detection.py")
        
        return marker_image_bgr
    else:
        # Use AprilTag generation library
        try:
            # Create tag generator
            tag_generator = apriltag(tag_family)
            
            # Generate tag
            tag_data = tag_generator.generate(tag_id)
            
            if tag_data is None:
                raise ValueError(f"Failed to generate tag ID {tag_id} for family {tag_family}")
            
            # Convert tag data to image
            # AprilTag data is typically a 2D array of 0s and 1s
            tag_array = np.array(tag_data, dtype=np.uint8)
            
            # Scale up the tag
            # Add border
            border_size = border_bits
            bordered_size = tag_array.shape[0] + 2 * border_size
            bordered_tag = np.zeros((bordered_size, bordered_size), dtype=np.uint8)
            bordered_tag[border_size:-border_size, border_size:-border_size] = tag_array * 255
            
            # Resize to desired output size
            tag_image = cv2.resize(bordered_tag, (size_px, size_px), interpolation=cv2.INTER_NEAREST)
            
            # Convert to 3-channel for saving
            tag_image_bgr = cv2.cvtColor(tag_image, cv2.COLOR_GRAY2BGR)
            
            # Save if output path provided
            if output_path:
                cv2.imwrite(str(output_path), tag_image_bgr)
                print(f"Generated AprilTag image: {output_path}")
                print(f"  Tag ID: {tag_id}")
                print(f"  Family: {tag_family}")
                print(f"  Size: {size_px}x{size_px} pixels")
            
            return tag_image_bgr
        except Exception as e:
            print(f"ERROR: AprilTag generation failed: {e}")
            print("Falling back to ArUco markers...")
            return generate_apriltag_image(tag_id, tag_family, size_px, border_bits, output_path, use_aruco=True)


def generate_multiple_tags(tag_ids, tag_family='tag36h11', size_px=200, output_dir='apriltags'):
    """
    Generate multiple AprilTag images
    
    Args:
        tag_ids: List of tag IDs to generate
        tag_family: Tag family
        size_px: Output image size
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"Generating {len(tag_ids)} AprilTag images...")
    print(f"Output directory: {output_path}")
    print()
    
    for tag_id in tag_ids:
        output_file = output_path / f"apriltag_{tag_family}_{tag_id:03d}.png"
        try:
            generate_apriltag_image(
                tag_id=tag_id,
                tag_family=tag_family,
                size_px=size_px,
                output_path=output_file,
                use_aruco=False  # Can be made configurable
            )
        except Exception as e:
            print(f"ERROR: Failed to generate tag ID {tag_id}: {e}")
    
    print()
    print(f"Generated {len(tag_ids)} tags in {output_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Generate AprilTag images')
    parser.add_argument('--tag-id', type=int, default=0,
                       help='Tag ID to generate (default: 0)')
    parser.add_argument('--tag-family', type=str, default='tag36h11',
                       help='Tag family (default: tag36h11)')
    parser.add_argument('--size', type=int, default=200,
                       help='Output image size in pixels (default: 200)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path (default: apriltag_{tag_id}.png)')
    parser.add_argument('--multiple', type=int, nargs='+', default=None,
                       help='Generate multiple tags (provide tag IDs as arguments)')
    parser.add_argument('--output-dir', type=str, default='apriltags',
                       help='Output directory for multiple tags (default: apriltags)')
    parser.add_argument('--aruco', action='store_true',
                       help='Use ArUco markers instead of AprilTag')
    
    args = parser.parse_args()
    
    if args.multiple:
        # Generate multiple tags
        generate_multiple_tags(
            tag_ids=args.multiple,
            tag_family=args.tag_family,
            size_px=args.size,
            output_dir=args.output_dir
        )
    else:
        # Generate single tag
        if args.output is None:
            args.output = f"apriltag_{args.tag_family}_{args.tag_id:03d}.png"
        
        try:
            generate_apriltag_image(
                tag_id=args.tag_id,
                tag_family=args.tag_family,
                size_px=args.size,
                output_path=args.output,
                use_aruco=args.aruco
            )
            print(f"\nTag generated successfully!")
            print(f"Print this image and measure the tag size (edge between white/black border)")
            print(f"Use --tag-size parameter in test_apriltag_detection.py with the measured size in meters")
        except Exception as e:
            print(f"ERROR: {e}")
            sys.exit(1)


if __name__ == '__main__':
    main()

