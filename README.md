# peoplenet_torch
Running PeopleNet through PyTorch
 
# installation
`pip install peoplenet_torch`

# usage
use the following code to get the bounding boxes of people from the image
```python
from peoplenet_torch import PeopleNet

model = PeopleNet()
boxes = model(image)
```