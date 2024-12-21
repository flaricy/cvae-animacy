from collections import abc 
import numpy as np
import torch 
from pathlib import Path

def is_seq_of(seq, expected_type, seq_type=None):
    if seq_type is None:
        exp_seq_type = abc.Sequence 
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
        
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True

class AverageHandler(object):
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.total = 0
        self.count = 0
        
    def update(self, value, n=1):
        self.total += value
        self.count += n
        
    def get_average(self):
        return self.total / self.count
    
def generate_html_with_videos(folder_path, output_html="index.html"):
    # List all video files in the folder
    video_extensions = [".mp4", ".avi", ".mov", ".wmv", ".flv"]
    folder = Path(folder_path)
    videos = [f for f in folder.iterdir() if f.suffix.lower() in video_extensions]

    # Generate HTML content
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Video Gallery</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f4f4f4;
            }
            .video-gallery {
                display: flex;
                overflow-x: auto;
                white-space: nowrap;
                padding: 20px;
                background-color: #333;
            }
            .video-gallery video {
                margin: 0 10px;
                border: 2px solid #fff;
                border-radius: 5px;
            }
            .video-gallery::-webkit-scrollbar {
                height: 8px;
            }
            .video-gallery::-webkit-scrollbar-thumb {
                background: #888;
                border-radius: 4px;
            }
            .video-gallery::-webkit-scrollbar-thumb:hover {
                background: #555;
            }
        </style>
    </head>
    <body>
        <div class="video-gallery">
    """

    # Add video elements
    for video in videos:
        video_path = video.resolve()
        html_content += f'            <video controls width="320" height="240" src="{video_path}" title="{video.name}"></video>\n'

    # Close HTML tags
    html_content += """
        </div>
    </body>
    </html>
    """

    # Save HTML to file
    output_file = Path(output_html)
    output_file.write_text(html_content, encoding="utf-8")

    print(f"HTML file generated: {output_html}")