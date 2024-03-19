# sample_seg

Explore basic sample-based segmentation methods

How to convert jpg to png (useful for labeling):
```
for img in *.jpg; do convert $img "${img%.jpg}.png"; done
```

How to run tests:
```
pytest tests/
```