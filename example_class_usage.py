import cv2
from DarkflowModel import DarkflowModel


#####################################
## EXAMPLE 1: LOAD FROM .weights FILE
#####################################
# dm = DarkflowModel(model_config='this_config.cfg',
#                    labels='./these_labels.txt',
#                    threshold=0.8,
#                    weights= 'these_weights.weights',
#                    construct=True)


######################################
## EXAMPLE 1: LOAD FROM .pb .meta FILE
######################################
# dm = DarkflowModel(model_config='this_config.cfg',
#                    labels='./these_labels.txt',
#                    threshold=0.8,
#                    model_pb='this_protobuf_file.pb',
#                    model_meta='this_meta_file.meta',
#                    construct=True)



########################################
## EXAMPLE 1: LOAD FROM checkpoint FILES
########################################
dm = DarkflowModel(model_config='this_config.cfg',
                   labels='./these_labels.txt',
                   threshold=0.8,
                   checkpoint=2685,
                   construct=True)


print(dm.id_dict)

test_image = cv2.imread("/path/to/image")
detections = dm.infer_one(im=test_image)
print(type(test_image))
for d in detections:
    print(d)
