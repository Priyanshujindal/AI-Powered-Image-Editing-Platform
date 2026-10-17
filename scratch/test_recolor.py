import os, sys, toml, numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ai

secrets = toml.load('.streamlit/secrets.toml')
os.environ['GEMINI_API_KEY'] = secrets['GEMINI_API_KEY']

# Mock login screen with green button
img = Image.new('RGB', (300, 400), (25, 27, 31))
arr = np.array(img)
arr[300:340, 30:270] = [0, 190, 110]
img = Image.fromarray(arr)

prompt = 'change the sign in button background from green to light blue'

# Step 1: Does keyword matcher wrongly trigger?
direct = ai.match_direct_edit(prompt, img)
if direct is None:
    print("Keyword match: NONE (correct - not a false positive!)")
else:
    print("Keyword match: FALSE POSITIVE:", direct[1])

# Step 2: Full edit_image
out, note = ai.edit_image(img, prompt)
px = out.getpixel((150, 320))
bg = out.getpixel((5, 5))
print(f"Button pixel: {px}")
print(f"Background pixel: {bg}")
print("Note:", repr(note[:100]))
