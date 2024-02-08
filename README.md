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

How to run tests and view the printed output to see where visualizations are saved:
```
pytest tests/ -s
```

How to run tests including slow animations
```
pytest tests/ -s --runslow
```
