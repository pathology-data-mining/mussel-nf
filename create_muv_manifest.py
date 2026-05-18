#!/usr/bin/env python3
"""
Create a samples CSV manifest for images in an S3 bucket.
Edit the `main()` function to configure your bucket and prefix.
"""

import subprocess
import csv
import re
from pathlib import Path

def get_s3_files(bucket, prefix, profile):
    """Get list of files from S3"""
    cmd = [
        'aws', 's3', 'ls',
        f's3://{bucket}/{prefix}',
        '--profile', profile,
        '--recursive'
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout.strip().split('\n')

def parse_s3_line(line):
    """Parse a line from aws s3 ls output"""
    # Format: date time size path
    parts = line.split(None, 3)
    if len(parts) < 4:
        return None

    s3_path = parts[3]
    return s3_path

def extract_metadata(s3_path):
    """Extract metadata from S3 path"""
    # Path format: external-data/muv/v1.0/{diagnosis}/{filename}
    path_parts = s3_path.split('/')

    if len(path_parts) < 5:
        return None

    filename = path_parts[4]

    # Extract sample ID (filename without extension)
    sample_id = Path(filename).stem

    return {
            'slide_id': sample_id,
            'slide_path': f's3://{bucket}/{s3_path}'
        }

def main():
    bucket = 'your-s3-bucket'
    prefix = 'external-data/muv/'
    profile = 'default'
    output_file = 'muv_samples.csv'

    print(f"Fetching files from s3://{bucket}/{prefix}...")
    lines = get_s3_files(bucket, prefix, profile)

    samples = []
    for line in lines:
        if not line.strip():
            continue

        s3_path = parse_s3_line(line)
        if not s3_path:
            continue

        metadata = extract_metadata(s3_path)
        if metadata:
            samples.append(metadata)

    print(f"Found {len(samples)} samples")

    # Write to CSV
    if samples:
        fieldnames = ['slide_id', 'slide_path']

        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(samples)

        print(f"Wrote manifest to {output_file}")

if __name__ == '__main__':
    main()
