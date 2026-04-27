Image 1 — 4×5
The × is being merged with 4 into one box (box 0), and 5 is box 1. Only 2 boxes instead of 3.
Why: The × strokes are touching or overlapping the 4. They share pixels so OpenCV sees them as one connected blob.
Fix — increase spacing when drawing. The × symbol naturally has diagonal strokes that bleed into neighbors. When you redraw this image, leave a bigger gap between 4 and × and between × and 5. That's the only reliable fix at this stage — the spec explicitly says:

Do NOT handle touching digits yet. Start with clean spaced symbols first.


Image 2 — 27÷2
The ÷ symbol is being split into 3 separate boxes (boxes 2, 3, 4, 5) — the top dot, the line, and the bottom dot are detected as individual contours.
Why: The ÷ has 3 disconnected parts. OpenCV sees 3 separate blobs because there's no pixel connecting them.
Two fixes to try:
Fix A — raise min_area to kill the dots:
pythonboxes, crops, binary = segment_expression(image_path, min_area=300)
The dots are tiny — if their area is below your threshold they disappear. Check what size they are in your debug output and set min_area just above that.
Fix B — avoid ÷ in test images for now. Replace it with / which is a single connected stroke and much easier to segment. Come back to ÷ later.