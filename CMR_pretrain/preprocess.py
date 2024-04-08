import os 
import numpy as np
import sympy as sp
from pydicom import dicomio
import joblib
import cv2
import scipy.ndimage

def create_person(data_path, save_path):
    '''
    example：1.pkl   {'SAX':[[], [], ...], '2ch':[[]], '4ch':[[]]}
    '''
    pid_list = os.listdir(data_path)
    pid_list.sort(key = lambda x: int(x))
    all_results = {}  
    sax_number = []
    for pid in pid_list:

        path1_case = os.path.join(data_path, pid, "study")  
        series_list = os.listdir(path1_case) 
        series_list.sort()

        a1 = a2 = a3 = 0
        dicom_axis_short = []
        dicom_axis_2ch = []
        dicom_axis_4ch = []
        for series in series_list:  

            path2_series = os.path.join(path1_case, series)  
            axis = series[0:3] 
            dicom_img = []
            dicom_list = os.listdir(path2_series) 
            dicom_list.sort()
            if axis == "sax":
                dicom_axis_short.append([os.path.join(path2_series, p) for p in dicom_list])
                a1 += 1
            elif axis == "2ch":
                dicom_axis_2ch.append([os.path.join(path2_series, p) for p in dicom_list])
                a2 += 1
            elif axis == "4ch":
                dicom_axis_4ch.append([os.path.join(path2_series, p) for p in dicom_list])
                a3 += 1
        print(f'case number:{pid}, SAX:{a1}, 2ch:{a2}, 4ch{a3}')
        sax_number.append(a1)

        dicom_person = {'SAX': dicom_axis_short, '2ch': dicom_axis_2ch, '4ch': dicom_axis_4ch}
        all_results[pid] = dicom_person

    return all_results, sax_number


def getinfo(img_file):
    RefDs = dicomio.read_file(img_file)
    img_array = RefDs.pixel_array
    ImagePosition = np.array(RefDs.ImagePositionPatient)
    ImageOrientation = np.array(RefDs.ImageOrientationPatient)
    PixelSpacing = RefDs.PixelSpacing
    SliceThickness = RefDs.SliceThickness
    normalvector = np.cross(ImageOrientation[0:3], ImageOrientation[3:6])
    TriggerTime = RefDs.TriggerTime
    SeriesNumber = RefDs.SeriesNumber
    
    return img_array, ImagePosition, PixelSpacing, ImageOrientation, SliceThickness, normalvector


def TO2D(s,eq,ImagePosition,ImageOrientationX,ImageOrientationY,PixelSpacing):

    x, y, z = sp.symbols('x, y, z')
    # check s  list out of range
    if s:
        x1_3d = s[0][0]
        y1_3d = s[0][1]
        pos = [x1_3d,y1_3d,z]
        differ = pos - ImagePosition
        differ_x = np.dot(differ,ImageOrientationX)
        differ_y = np.dot(differ,ImageOrientationY)
        pos_2d = [differ_x/PixelSpacing[0], differ_y/PixelSpacing[1]]
        return pos_2d
    else:
        s = list(sp.linsolve(eq, [x, z]))
        x1_3d = s[0][0]
        z1_3d = s[0][1]
        pos = [x1_3d, y, z1_3d]
        differ = pos - ImagePosition
        differ_x = np.dot(differ, ImageOrientationX)
        differ_y = np.dot(differ, ImageOrientationY)
        pos_2d = [differ_x/PixelSpacing[0], differ_y/PixelSpacing[1]]
        print("s is ERROR")
        return pos_2d
    

def get_2dpos(pos_2d, img_array):
    x, y, z = sp.symbols('x, y, z')
    exp_x = sp.solve([x-pos_2d[0], y-pos_2d[1]], [x, z])[x]

    x1 = sp.solve([x - exp_x, y], [x, y])[x]  # 取出y=0的时候的x值
    x2 = sp.solve([x - exp_x, y - img_array.shape[0]], [x, y])[x]  # 取出y=256的时候的x值
    return x1, x2


def res_line(img_array, x1, x2):
    pos = [x1, x2]

    heat_map1 = np.zeros((img_array.shape[0],img_array.shape[1], 3), dtype="uint8")
    cv2.line(heat_map1, (pos[0],0), (pos[1],img_array.shape[0]), (0,0,255), lineType=cv2.LINE_AA)  #
    img_line = cv2.cvtColor(heat_map1, cv2.COLOR_RGB2GRAY)

    return  img_line


def CenterLabelHeatMap(img_array1, point1, point2, sigma, scale=1):

    heat_map1 = np.zeros((img_array1.shape[0], img_array1.shape[1], 3), dtype="uint8")
    img_height = img_array1.shape[0]
    img_width = img_array1.shape[1]
    cv2.line(heat_map1, point1,point2, (1, 0, 0), lineType=cv2.LINE_AA)
    lineY1 = point1[1]
    lineY2 = point2[1]
    # A=Y2-Y1    B=X1-X2     C=X2*Y1-X1*Y2
    a = lineY2 - lineY1
    b = point1[0] -point2[0]
    c = point2[0]*lineY1-point1[0]*lineY2

    X1 = np.linspace(1, img_width, img_width)
    Y1 = np.linspace(1, img_height, img_height)
    [X, Y] = np.meshgrid(X1, Y1)

    dis = (pow((a*X + b*Y + c), 2)) / (a*a + b*b)
    exponent = dis / sigma / sigma / 2
    heatmap =scale * np.exp(-exponent.astype(float))

    return heatmap


def resample_sapcing(image,spacing, new_spacing=[1.26,1.26]):
    # Determine current pixel spacing
    spacing = np.array(list(spacing))
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)

    return image


def get_square_crop(img, base_size, crop_size):
    try:
        res = img
        height, width = res.shape
    except:
        pass
    else:
        if height < base_size[0]:
            diff = base_size[0] - height
            extend_top = int(diff / 2)
            dx_odd= int(extend_top % 2 == 1)
            extend_bottom = int(diff - extend_top)
            res = cv2.copyMakeBorder(res, extend_top+dx_odd, extend_bottom, 0, 0, borderType=cv2.BORDER_CONSTANT, value=0)#是边框都填充成一个固定值，比如下面的程序都填充0
            height = base_size[0]

        if width < base_size[1]:
            diff = base_size[1] - width
            extend_top = int(diff / 2)
            dy_odd = int(extend_top % 2 == 1)
            extend_bottom = int(diff - extend_top)
            res = cv2.copyMakeBorder(res, 0, 0, extend_top+dy_odd, extend_bottom, borderType=cv2.BORDER_CONSTANT, value=0)
            width = base_size[1]

    crop_y_start = (height - crop_size[0]) / 2
    crop_x_start = (width - crop_size[1]) / 2
    res = res[int(crop_y_start):int(crop_y_start + crop_size[0]), int(crop_x_start):int(crop_x_start + crop_size[1])]
    return res


def zero(img):

    img = np.float32(img)
    img -= img.mean()
    img /= img.std()
    img = (img.reshape(1, img.shape[0], img.shape[1]))
    return img


def cal_heatmap(sax_path, lax_path, crop_size):

    sax = sax_path[0] 
    lax = lax_path[0] 
    
    img_array1, ImagePosition1, PixelSpacing1, ImageOrientation1, SliceThickness1,normalvector1 = getinfo(sax)
    img_array2, ImagePosition2, PixelSpacing2, ImageOrientation2, SliceThickness2,normalvector2 = getinfo(lax)
    ImageOrientationX1 = ImageOrientation1[0:3]
    ImageOrientationY1 = ImageOrientation1[3:6]
    
    sp.init_printing(use_unicode=True)
    x, y, z = sp.symbols('x, y, z')
    
    eq = [normalvector1[0] * (x - ImagePosition1[0]) + normalvector1[1] * (y - ImagePosition1[1]) + normalvector1[2] * (z - ImagePosition1[2]),
          normalvector2[0] * (x - ImagePosition2[0]) + normalvector2[1] * (y - ImagePosition2[1]) + normalvector2[2] * (z - ImagePosition2[2])]
    s = list(sp.linsolve(eq, [x, y]))
    #
    pos_2d = TO2D(s, eq, ImagePosition1, ImageOrientationX1, ImageOrientationY1, PixelSpacing1)
    x1_1, x1_2 = get_2dpos(pos_2d, img_array1)
    
    heatmap = res_line(img_array1, x1_1, x1_2)
    heatmap = resample_sapcing(heatmap, PixelSpacing1, new_spacing=[1.26,1.26])
    heatmap = get_square_crop(heatmap, crop_size, crop_size)
    # 
    point = np.where(heatmap==np.max(heatmap))
    if len(point[0])==1:
        point = np.where(heatmap == np.max(heatmap)-1)
    point1 = (point[1][0], point[0][0])
    point2 = (point[1][-1], point[0][-1])
    
    heatmap = CenterLabelHeatMap(heatmap, point1 ,point2, 6.0, scale=1)
    heatmap = heatmap.reshape(1, heatmap.shape[0], heatmap.shape[1])
    Slice_loc = np.dot((normalvector1/np.linalg.norm(normalvector1)),ImagePosition1)
    
    return heatmap, -Slice_loc


def pro_img(img_path, crop_size):

    img_array, _, PixelSpacing, _, _, _ = getinfo(img_path)
    img = resample_sapcing(img_array, PixelSpacing, new_spacing=[1.26,1.26])
    img = get_square_crop(img, crop_size, crop_size)
    img = np.uint8(np.array(img))
    img = zero(img)
    return img


def create_data_select_multi(data_path, image_size=np.array([224, 224])):

    case_cur = data_path
    crop_size = image_size
    data_case_all = []  # 
    for num_s in range(len(case_cur['SAX'])):  
        #
        heatmap_truex1, Slice_loc = cal_heatmap(case_cur['SAX'][num_s], case_cur['2ch'][0], crop_size)
        heatmap_truex2, Slice_loc = cal_heatmap(case_cur['SAX'][num_s], case_cur['4ch'][0], crop_size)
        heatmap_true = np.vstack((heatmap_truex1,heatmap_truex2))
        ######################################################
        SAX_images = []
        SAX_paths = case_cur['SAX'][num_s]
        for num_t in range(len(case_cur['SAX'][num_s])):
            # 
            SAX_image = pro_img(case_cur['SAX'][num_s][num_t], crop_size)
            SAX_images.append(SAX_image)
        # 
        data_case = {'image': SAX_images, 'image_path': SAX_paths, 'heatmap': heatmap_true, 'slice_loc': Slice_loc}  # Slice_loc是计算出的层
        data_case_all.append(data_case)
        
    slices = len(data_case_all) 
    SeriesNumber_set2 = [data_case_all[slice_id]['slice_loc'] for slice_id in range(slices)]
    Slices_loc = np.argsort(SeriesNumber_set2) 

    for slice_id in range(slices):
        data_case_all[slice_id]['rel_gap'] = SeriesNumber_set2[slice_id] - SeriesNumber_set2[Slices_loc[0]]   #slice的位置(相对位置) #当前series的slice_loc - 最小series的slice_loc
        data_case_all[slice_id]['abs_gap'] = abs(SeriesNumber_set2[Slices_loc[0]] - SeriesNumber_set2[Slices_loc[-1]]) #loc的总体值
        
    return data_case_all


def create_data(data_ori_path, save_data_path, filter_set):

    pid_list = data_ori_path.keys()
    num = 0 
    for pid in pid_list: 
        if pid in filter_set:
            num += 1 
            save_data = create_data_select_multi(data_ori_path[pid])
            save_path = os.path.join(save_data_path, '{}.pkl'.format(pid))
            with open(save_path, 'wb') as f:
                joblib.dump(save_data, f, protocol=4)

    print('done num: ', num)


if __name__ '__main__':
	phase = 'train'
	data_path = './data/DSBCC/{}/'.format(phase)
	save_path = './data/DSBCC_processed/{}/'.format(phase)
	fname_train = './select_person/{}_person_id.txt'.format(phase)
	with open(fname_train, encoding="utf8", errors='ignore') as f:
    	train_set = [i[:-1].split(' ') for i in f.readlines()][0]

	all_results, sax_number = create_person(dataorig_path, datasave_path)
	create_data(all_results, save_path, filter_set=train_set)