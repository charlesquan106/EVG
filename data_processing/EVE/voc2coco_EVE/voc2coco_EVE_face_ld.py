import xml.etree.ElementTree as ET
import os
import json
from pathlib import Path
import argparse

coco = dict()
coco['images'] = []
coco['type'] = 'point'
coco['annotations'] = []

category_set = dict()
image_set = set()

category_item_id = 0
image_id = 100000000000
annotation_id = 0

def addCatItem(name):
    global category_item_id
    category_item = dict()
    category_item['supercategory'] = 'none'
    category_item_id += 1
    category_item['id'] = category_item_id
    category_item['name'] = name
    coco['categories'].append(category_item)
    category_set[name] = category_item_id
    return category_item_id

def addImgItem(file_name, size):
    global image_id
    if file_name is None:
        raise Exception('Could not find filename tag in xml file.')
    if size['width'] is None:
        raise Exception('Could not find width tag in xml file.')
    if size['height'] is None:
        raise Exception('Could not find height tag in xml file.')
    image_id += 1
    image_item = dict()
    image_item['id'] = image_id
    image_item['file_name'] = file_name
    image_item['width'] = size['width']
    image_item['height'] = size['height']
    coco['images'].append(image_item)
    image_set.add(file_name)
    return image_id


def addAnnoItem(image_id, gazepoint_list, gazeOrigin_list, gazeTarget_list,\
            camera_list, cameraTransformation_list, headRvec_list, gazeR_list, monitorPose_list, screenSize_list,faceBbox_list):
    global annotation_id
    # , cameraMatrix, distCoeffs, monitorPose
    annotation_item = dict()
    annotation_item['image_id'] = image_id
    annotation_item['gazepoint'] = gazepoint_list
    annotation_item['gazeOrigin'] = gazeOrigin_list
    annotation_item['gazeTarget'] = gazeTarget_list
    annotation_item['camera'] = camera_list
    annotation_item['cameraTransformation'] = cameraTransformation_list
    # print(annotation_item['cameraTransformation'])
    annotation_item['headRvec'] = headRvec_list
    annotation_item['gazeR'] = gazeR_list
    annotation_item['monitorPose'] = monitorPose_list
    annotation_item['screenSize'] = screenSize_list
    annotation_item['faceBbox'] = faceBbox_list
    annotation_id += 1
    annotation_item['id'] = annotation_id

    coco['annotations'].append(annotation_item)

def parseXmlFiles(xml_path):
    for f in os.listdir(xml_path):
        if not f.endswith('.xml'):
            continue

        real_file_name = f.split(".")[0] + ".jpg"
        gazepoint_dict = dict()
        gazeOrigin_dict = dict()
        gazeTarget_dict = dict()
        
        cameraTransformation_dict = dict()
        headRvec_dict = dict()
        gazeR_dict = dict()
        
        faceBbox_dict = dict()
        size = dict()
        current_image_id = None
        file_name = None
        size['width'] = None
        size['height'] = None
        size['depth'] = None
        
        camera_dict = {
            'cameraMatrix': [],
            'distCoeffs': []
            }
        monitorPose_dict = {
            'rvects': [],
            'tvecs': []
            }
        screenSize_dict = {
            'unit_mm': [],
            'unit_pixel': []
            }

        xml_file = os.path.join(xml_path, f)
        # print(xml_file)

        tree = ET.parse(xml_file)
        root = tree.getroot()
        if root.tag != 'annotation':
            raise Exception(
                'pascal voc xml root element should be annotation, rather than {}'
                .format(root.tag))

        #elem is <folder>, <filename>, <size>, <object>, <gazepoint>
        for elem in root:
            current_parent = elem.tag
            current_sub = None

            if elem.tag == 'folder':
                continue

            if elem.tag == 'filename':
                file_name = real_file_name  #elem.text
                if file_name in category_set:
                    raise Exception('file_name duplicated')

            #add img item only after parse <size> tag
            elif current_image_id is None and file_name is not None and size[
                    'width'] is not None:
                # # print(file_name, "===", image_set)
                if file_name not in image_set:
                    current_image_id = addImgItem(file_name, size)
                    # print('add image with {} and {}'.format(file_name, size))
                else:
                    pass
                    # raise Exception('duplicated image: {}'.format(file_name))
                    
                    
            #subelem is <width>, <height>, <depth>, <name>, <bndbox>, <gazep>
            
            
            
            gazepoint_dict['gp_x'] = None
            gazepoint_dict['gp_y'] = None
            gazeOrigin_dict['gaze_origin'] = None
            gazeTarget_dict['gaze_target'] = None
            faceBbox_dict['face_bbox'] = None
            
            cameraTransformation_dict['camera_transformation'] = None
            headRvec_dict['head_rvec'] = None
            gazeR_dict['gaze_R'] = None
            

            gazepoint_list = []
            gazeOrigin_list = []
            gazeTarget_list = []
            faceBbox_list = []
            camera_list = []
            monitorPose_list = []
            screenSize_list = []
            cameraTransformation_list  = []
            headRvec_list = []
            gazeR_list = []
            


            
            for subelem in elem:
            
                obejct_done = False
                
                # print(subelem)
                current_sub = subelem.tag
                # print(current_sub)

                if current_parent == 'size':
                    #subelem is <width>, <height>, <depth> when current_parent is <size>
                    if size[subelem.tag] is not None:
                        raise Exception('xml structure broken at size tag.')
                    size[subelem.tag] = int(subelem.text)
                elif current_parent == 'object':

                    #subsubelem is <gp_x>, <gp_y>, when subelem is <gazepoint>
                    if current_sub == 'gazepoint':
                        for option in subelem:
                            if gazepoint_dict[option.tag] is not None:
                                raise Exception('xml structure corrupted at gazepoint tag.')
                            gazepoint_dict[option.tag] = int(option.text)
                        # print(gazepoint_dict)
                        
                        if gazepoint_dict['gp_x'] is None :
                            raise Exception('xml structure broken at gazepoint')
                        else:
                            #gazepoint
                            gazepoint_list.append(gazepoint_dict['gp_x'])
                            gazepoint_list.append(gazepoint_dict['gp_y'])
                            
                    #subsubelem is <gaze_origin>, when subelem is <gazeOrigin>
                    elif current_sub == 'gazeOrigin':
                        for option in subelem:
                            if gazeOrigin_dict[option.tag] is not None:
                                raise Exception('xml structure corrupted at gazeOrigin tag.')
                            gazeOrigin_dict[option.tag] = option.text
                        # print(gazeOrigin_dict)
                        
                        if gazeOrigin_dict['gaze_origin'] is None :
                            raise Exception('xml structure broken at gazeOrigin')
                        else:
                            #gazepoint
                            gazeOrigin_list.append(gazeOrigin_dict['gaze_origin'])
                            
                    #subsubelem is <gaze_target>, when subelem is <gazeTarget>        
                    elif current_sub == 'gazeTarget':
                        for option in subelem:
                            if gazeTarget_dict[option.tag] is not None:
                                raise Exception('xml structure corrupted at gazeTarget tag.')
                            gazeTarget_dict[option.tag] = option.text
                        # print(gazeTarget_dict)
                        
                        if gazeTarget_dict['gaze_target'] is None :
                            raise Exception('xml structure broken at gazeTarget')
                        else:
                            #gazeTarget
                            gazeTarget_list.append(gazeTarget_dict['gaze_target'])
                            
                    #option is <cameraMatrix> <distCoeffs>, when subelem is <camera>
                    elif current_sub == 'camera':     
                        for option in subelem:
                            camera_dict[option.tag] = option.text  
                        # print(camera_dict)
                        
                        if camera_dict['cameraMatrix'] is None :
                            raise Exception('xml structure broken at camera')
                        else:
                            #camera
                            camera_list.append(camera_dict['cameraMatrix'])
                            camera_list.append(camera_dict['distCoeffs'])
                            
                    #subsubelem is <camera_transformation>, when subelem is <cameraTransformation>
                    elif current_sub == 'cameraTransformation':
                        import torch
                        for option in subelem:
                            if cameraTransformation_dict[option.tag] is not None:
                                raise Exception('xml structure corrupted at cameraTransformation tag.')
                            cameraTransformation_dict[option.tag] = option.text

                            # print(cameraTransformation_dict[option.tag])
                            # print(option.tag)
                            # print(option.text)
                            # cameraTransformation_dict[option.tag] =  option.text
                            # cameraTransformation_dict[camera_transformation]
                            # print(type(cameraTransformation_dict[option.tag]))
                            # print(cameraTransformation_dict[option.tag])
                        
                        if cameraTransformation_dict['camera_transformation'] is None :
                            raise Exception('xml structure broken at cameraTransformation')
                        else:
                            #cameraTransformation
                            # print(type(cameraTransformation_dict[option.tag]))
                            cameraTransformation_list.append(cameraTransformation_dict[option.tag])
                            # print(type(cameraTransformation_list[0]))
                            
                                        #subsubelem is <camera_transformation>, when subelem is <cameraTransformation>
                    elif current_sub == 'headRvec':
                        for option in subelem:
                            if headRvec_dict[option.tag] is not None:
                                raise Exception('xml structure corrupted at headRvec tag.')
                            headRvec_dict[option.tag] = option.text
                        
                        if headRvec_dict['head_rvec'] is None :
                            raise Exception('xml structure broken at headRvec')
                        else:
                            #headRvec
                            headRvec_list.append(headRvec_dict['head_rvec'])
                            
                    elif current_sub == 'gazeR':
                        for option in subelem:
                            if gazeR_dict[option.tag] is not None:
                                raise Exception('xml structure corrupted at gazeR tag.')
                            gazeR_dict[option.tag] = option.text
                        
                        if gazeR_dict['gaze_R'] is None :
                            raise Exception('xml structure broken at gazeR')
                        else:
                            #gazeR
                            gazeR_list.append(gazeR_dict['gaze_R'])
                            
                    #option is <rvects> <tvecs>, when subelem is <monitorPose>
                    elif current_sub == 'monitorPose':     
                        for option in subelem:
                            monitorPose_dict[option.tag] = option.text  
                        # print(monitorPose_dict)
                        
                        if monitorPose_dict['rvects'] is None :
                            raise Exception('xml structure broken at monitorPose')
                        else:
                            #monitorPose
                            monitorPose_list.append(monitorPose_dict['rvects'])
                            monitorPose_list.append(monitorPose_dict['tvecs'])
                            
                    #option is <unit_mm>, <unit_pixel>, when subelem is <landmark>
                    elif current_sub == 'screenSize':     
                        for option in subelem:
                            screenSize_dict[option.tag] = option.text  
                        # print(screenSize_dict)
                        
                        if screenSize_dict['unit_mm'] is None :
                            raise Exception('xml structure broken at gazepoint')
                        else:
                            #screenSize
                            screenSize_list.append(screenSize_dict['unit_mm'])
                            screenSize_list.append(screenSize_dict['unit_pixel'])
                            
                    elif current_sub == 'faceBbox':
                        for option in subelem:
                            if faceBbox_dict [option.tag] is not None:
                                raise Exception('xml structure corrupted at faceBbox tag.')
                            faceBbox_dict[option.tag] = option.text
                        # print(gazeTarget_dict)
                        
                        if faceBbox_dict['face_bbox'] is None :
                            raise Exception('xml structure broken at faceBbox')
                        else:
                            #gazeTarget
                            faceBbox_list.append(faceBbox_dict['face_bbox'])

                            
                        obejct_done = True
                        
                #only after parse the <object> tag
                if obejct_done:
                    if current_image_id is None:
                        raise Exception('xml structure broken at current_image_id')
                
                    
                    # print('add annotation with {},{},{}'.format(
                        # current_image_id,gazepoint_list,screenSize_list))
                    addAnnoItem( current_image_id, gazepoint_list, gazeOrigin_list, gazeTarget_list,\
                        camera_list, cameraTransformation_list, headRvec_list, gazeR_list, monitorPose_list, screenSize_list,faceBbox_list)
                    
def create_directory_if_not_exists(path):
    
    if not os.path.exists(path):
        os.makedirs(path)
        # print(f"Created directory: {path}")
        
        
        
def main():
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--datatype', type=str, required=True)
    args = parser.parse_args()
    datatype = args.datatype 
    
    # xml_path = f'/home/owenserver/storage/Datasets/VOC_format_EVE_Data_ld_s_test/{datatype}/annotation_xml'
    # json_dir = f'/home/owenserver/storage/Datasets/VOC_format_EVE_Data_ld_s_test/{datatype}/annotation'
    
    # xml_path = f'/home/owenserver/storage/Datasets/EVE/VOC_s_eve_dataset/{datatype}/annotation_xml'
    # json_dir = f'/home/owenserver/storage/Datasets/EVE/VOC_s_eve_dataset/{datatype}/annotation'
    # json_filename = f'gaze_EVE_ld_s_{datatype}.json'
    
    xml_path = f'/home/owenserver/storage/Datasets/VOC_format_EVE_Data_ld_ext/{datatype}/annotation_xml'
    json_dir = f'/home/owenserver/storage/Datasets/VOC_format_EVE_Data_ld_ext/{datatype}/annotation'
    
    # xml_path = f'/home/owenserver/storage/Datasets/VOC_format_EVE_{datatype}/{datatype}/annotation_xml'
    # json_dir = f'/home/owenserver/storage/Datasets/VOC_format_EVE_{datatype}/{datatype}/annotation'
    
    
    # xml_path = f'/home/owenserver/Python/Gaze_DataProcessing/EVE/VOC_format_EVE_Data_ld_ext/{datatype}/annotation_xml'
    # json_dir = f'/home/owenserver/Python/Gaze_DataProcessing/EVE/VOC_format_EVE_Data_ld_ext/{datatype}/annotation'
    
    json_filename = f'gaze_EVE_ld_ext_{datatype}.json'
    
    json_dir = Path(json_dir)
  
    json_path = json_dir / json_filename
    # print(json_path)
    
    create_directory_if_not_exists(json_dir)
    
    parseXmlFiles(xml_path)
    json.dump(coco, open(json_path, 'w'))

if __name__ == '__main__':
    main()