from Post_Process import post_processing

'''
===============================================================================================
Usable weight function: [normal, Laplacian, ML_Laplacian, MLsum]
multi label calculated: [normal, Laplacian] (ML_Laplacian, MLsum for multi label are not done.)
result image for multi label is not implemented(just IoUs and mIoU)
===============================================================================================
'''

### If you want to easily checking results images and IoUs....

weights = ['normal', 'Laplacian', 'ML_Laplacian', 'MLsum']
for weight in weights:
    post_processing(weight, is_multi=False, show=True)
for weight in weights[:2]:
    post_processing(weight,is_multi=True)

