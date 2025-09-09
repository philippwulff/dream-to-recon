```bash
ffmpeg -i 0to4499_BTS_D_waymo.mp4 -vf "crop=360:240:120:60" 0to4499_BTS_D_waymo_crop.mp4
ffmpeg -i reconstructions_waymo.mp4 -vf "crop=2870:480:0:0" reconstructions_waymo_crop.mp4
ffmpeg -i reconstructions_waymo_crop.mp4 -r 3 reconstructions_waymo_crop_3fps.mp4
ffmpeg -i reconstructions_waymo_crop_3fps.mp4 -filter:v "setpts=PTS/2" -an reconstructions_waymo_crop_6fps_2x.mp4

# Teaser
ffmpeg -i teaser.mp4 -vf "crop=3000:720:8:0" teaser_crop.mp4
ffmpeg -i teaser_crop.mp4 -r 3 teaser_crop_3fps.mp4
ffmpeg -i teaser_crop_3fps.mp4 -filter:v "setpts=PTS/3" -an teaser_crop_3fps_3x.mp4

# Occlusions
ffmpeg -i occlusions.mp4 -r 3 occlusions_3fps.mp4
ffmpeg -i occlusions_3fps.mp4 -filter:v "setpts=PTS/2" -an occlusions_3fps_2x.mp4

# NVS
 × 720
ffmpeg -i nvs_720p.mp4 -vf "crop=1620:720:8:0" nvs_720p_crop.mp4
ffmpeg -i nvs_720p_crop.mp4 -r 5 nvs_720p_crop_5fps.mp4
ffmpeg -i nvs_1080p.mp4 -r 5 nvs_1080p_5fps.mp4
ffmpeg -i nvs_480p.mp4 -r 5 nvs_480p_5fps.mp4
# ffmpeg -i occlusions_3fps.mp4 -filter:v "setpts=PTS/2" -an occlusions_3fps_2x.mp4
```