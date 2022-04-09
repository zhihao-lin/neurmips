python experts_test.py test.mode=render model.bake=True --config-name replica-apartment0
python experts_test.py test.mode=render model.bake=True --config-name replica-apartment1
python experts_test.py test.mode=render model.bake=True --config-name replica-apartment2
python experts_test.py test.mode=render model.bake=True --config-name replica-frl0
python experts_test.py test.mode=render model.bake=True --config-name replica-kitchen
python experts_test.py test.mode=render model.bake=True --config-name replica-room0
python experts_test.py test.mode=render model.bake=True --config-name replica-room2

python -m mnh.metric -rewrite output_images/experts/replica-apartment0/color/valid/ 
python -m mnh.metric -rewrite output_images/experts/replica-apartment1/color/valid/ 
python -m mnh.metric -rewrite output_images/experts/replica-apartment2/color/valid/ 
python -m mnh.metric -rewrite output_images/experts/replica-frl0/color/valid/ 
python -m mnh.metric -rewrite output_images/experts/replica-kitchen/color/valid/
python -m mnh.metric -rewrite output_images/experts/replica-room0/color/valid/ 
python -m mnh.metric -rewrite output_images/experts/replica-room2/color/valid/ 

