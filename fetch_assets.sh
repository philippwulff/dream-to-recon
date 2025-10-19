#!/usr/bin/env bash

# Usage: ./download_assets.sh {GT|checkpoints}

set -euo pipefail

[[ $# -eq 1 ]] || { echo "Usage: $0 {GT|checkpoints}"; exit 1; }

# Run from the script's directory so relative paths work
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "$1" in
  GT)
    echo "----------------------- Downloading LiDAR occupancy volumes for quantitative evaluation -----------------------"
    mkdir -p data

    # KITTI-360
    wget -O data/GT_Occ_kitti360.zip \
      "https://github.com/philippwulff/dream-to-recon/releases/download/v0.0.1/GT_Occ_kitti360.zip"
    unzip -o data/GT_Occ_kitti360.zip -d data/
    rm -f data/GT_Occ_kitti360.zip

    # Waymo
    wget -O data/GT_Occ_waymo.zip \
      "https://github.com/philippwulff/dream-to-recon/releases/download/v0.0.1/GT_Occ_waymo.zip"
    unzip -o data/GT_Occ_waymo.zip -d data/
    rm -f data/GT_Occ_waymo.zip
    ;;

  checkpoints)
    echo "----------------------- Downloading pretrained models -----------------------"

    # Match each LINK to its OUTPUT path (all are .zip)
    LINKS=(
      "https://github.com/philippwulff/dream-to-recon/releases/download/v0.0.1/BTS_kitti360.pt.zip"
      "https://github.com/philippwulff/dream-to-recon/releases/download/v0.0.1/BTS-D_kitti360.pt.zip"
      "https://github.com/philippwulff/dream-to-recon/releases/download/v0.0.1/Ours-Distilled_kitti360.pt.zip"
      "https://github.com/philippwulff/dream-to-recon/releases/download/v0.0.1/BTS_waymo.pt.zip"
      "https://github.com/philippwulff/dream-to-recon/releases/download/v0.0.1/BTS-D_waymo.pt.zip"
      "https://github.com/philippwulff/dream-to-recon/releases/download/v0.0.1/Ours-Distilled_waymo.pt.zip"
      TODO add controlnet
    )
    OUTPUTS=(
      "out/kitti360/pretrained/BTS_kitti360.pt.zip"
      "out/kitti360/pretrained/BTS-D_kitti360.pt.zip"
      "out/kitti360/pretrained/Ours-Distilled_kitti360.pt.zip"
      "out/waymo/pretrained/BTS_waymo.pt.zip"
      "out/waymo/pretrained/BTS-D_waymo.pt.zip"
      "out/waymo/pretrained/Ours-Distilled_waymo.pt.zip"
      TODO add controlnet
    )

    [[ ${#LINKS[@]} -eq ${#OUTPUTS[@]} ]] || { echo "LINKS/OUTPUTS length mismatch"; exit 2; }

    for i in "${!LINKS[@]}"; do
      download_link="${LINKS[$i]}"
      output_path="${OUTPUTS[$i]}"
      outdir="$(dirname "$output_path")"

      echo "Downloading \"$download_link\" -> \"$output_path\""
      mkdir -p "$outdir"
      wget -O "$output_path" "$download_link"

      # Always unzip (all checkpoints are .zip), then remove archive
      unzip -o "$output_path" -d "$outdir"
      rm -f "$output_path"
    done
    ;;

  *)
    echo "Invalid argument: '$1' (use 'GT' or 'checkpoints')" >&2
    exit 1
    ;;
esac
