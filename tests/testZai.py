from zai import ZaiClient
client = ZaiClient(api_key="15ff9c13c6f243b1b168e670d6f0056d.rY51n0plxTkvwaEF")

# Generate video
response = client.videos.generations(
    model="cogvideox-3",
    prompt="A cat is playing with a ball.",
    quality="quality",  # Output mode, "quality" for quality priority, "speed" for speed priority
    with_audio=True, # Whether to include audio
    size="1920x1080",  # Video resolution, supports up to 4K (e.g., "3840x2160")
    fps=30,  # Frame rate, can be 30 or 60
)
print(response)

# Get video result
result = client.videos.retrieve_videos_result(id=response.id)
print(result)