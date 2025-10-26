import cv2
import cv2.aruco as ar
import numpy as np

# generation
def generate_markers():
    dict = ar.getPredefinedDictionary(ar.DICT_6X6_250)
    image = np.zeros((250,250))
    for i in range(36):
        cv2.imwrite(f"marker{i}.png",ar.generateImageMarker(dict,i,200,1))


#detection
def detect_markers():
    # cap = cv2.VideoCapture(0)
    # cv2.namedWindow("image",cv2.WINDOW_GUI_NORMAL)
    image_path = './template_cut.png'

    detector_params = ar.DetectorParameters()
    dict = ar.getPredefinedDictionary(ar.DICT_6X6_250)
    detector = ar.ArucoDetector(dict,detector_params)
        
    # while True:
    # _,image = cap.read()
    image = cv2.imread(image_path)
    #image = cv2.GaussianBlur(image,(3,3),0)
    corners,ids,rejected=detector.detectMarkers(image)
    if ids is not None:
        print(corners[0][0][0])
        centers = []
        for corner in corners:
            for dot in corner[0]:
                centers.append(dot)
            #cv2.circle(image,center[0].astype(int),10,(255,0,0),5)

        centers = np.array(centers)

        rect = cv2.minAreaRect(centers)
        rect_corners = cv2.boxPoints(rect)
        print(rect_corners)
        # 9x6 (100+10)

        # for corner in rect_corners:
        #     cv2.circle(image,corner.astype(int),10,(0,0,255),5)

    if rect_corners is not None:
        cords = get_three_corners(rect_corners)
        warped = transform_board(image,cords)
        calculate_fields(warped,detector, corners)

    cv2.imshow('image',image)
    cv2.imshow('warped',warped)
    cv2.waitKey()
        # if cv2.waitKey(1)==ord('q'):
        #     break
    cv2.destroyAllWindows()


def calculate_fields(warped,detector,corners):

    corners,ids,rejected=detector.detectMarkers(warped)
    print(warped.shape)
    dots=np.array(find_closest_markers(corners,warped.shape[1],warped.shape[0]))
    
    first_column_boundaries = np.array([dots[0],dots[1]]).astype(np.float32).reshape(8,2)
    first_column = cv2.minAreaRect(first_column_boundaries)
    first_column_corners = cv2.boxPoints(first_column)
    cv2.drawContours(warped, [first_column_corners.astype(int)], 0, (0,255,0), 5)

    first_row_boundaries = np.array([dots[0],dots[2]]).astype(np.float32).reshape(8,2)
    first_row = cv2.minAreaRect(first_row_boundaries)
    first_row_corners = cv2.boxPoints(first_row)
    cv2.drawContours(warped, [first_row_corners.astype(int)], 0, (0,0,255), 5)

    cols,rows = 0,0

    for corner in corners:
        center = corner.mean(axis=1)[0]
        if cv2.pointPolygonTest(first_column_corners, center, False) > 0:
            point1, point2 = np.array([-1,-1]),np.array([-1,-1])
            for point in corner[0]:
                if point[1] > point1[1]:
                    point2 = point1
                    point1 = point
                if point[1] > point2[1] and point[0] != point1[0]:
                    point2 = point
            print(f'point {point1}x{point2}')
            cv2.line(warped,point1.astype(int),point2.astype(int),(255,0,0),5)
            rows += 1
        
        if cv2.pointPolygonTest(first_row_corners, center, False) > 0:
            point1, point2 = np.array([-1,-1]),np.array([-1,-1])
            for point in corner[0]:
                if point[0] > point1[0]:
                    point2 = point1
                    point1 = point
                if point[0] > point2[0] and point[1] != point1[1]:
                    point2 = point
            print(f'point {point1}x{point2}')
            cv2.line(warped,point1.astype(int),point2.astype(int),(255,0,0),5)
            cols += 1

    print(f'{cols}x{rows}')

    cv2.aruco.drawDetectedMarkers(warped,corners,ids)


def find_closest_markers(markers,height,width):
    upper_left=[0,0]
    upper_right=[0, width]
    bottom_left=[height, 0]
    
    upper_left_dist= width
    upper_right_dist=width
    bottom_left_dist=width

    upper_left_res= None
    upper_right_res=None
    bottom_left_res=None

    for marker in markers:
        center = marker.mean(axis=1)[0]

        upper_left_cur_dist= find_length(upper_left,center)
        upper_right_cur_dist=find_length(upper_right,center)
        bottom_left_cur_dist=find_length(bottom_left,center)
            
        if upper_left_cur_dist < upper_left_dist:
            upper_left_dist = upper_left_cur_dist
            upper_left_res = marker

        if upper_right_cur_dist < upper_right_dist:
            upper_right_dist = upper_right_cur_dist
            upper_right_res = marker
        
        if bottom_left_cur_dist < bottom_left_dist:
            bottom_left_dist = bottom_left_cur_dist
            bottom_left_res = marker

    return [upper_left_res, upper_right_res, bottom_left_res]

def get_three_corners(corners):
    return np.array([corners[1],corners[0],corners[2]])
    # return np.array([corners[1],corners[0],corners[2],corners[3]])
        
def find_length(start,dest):
    return np.sqrt((start[0]-dest[0])**2+(start[1]-dest[1])**2)

def transform_board(image, corners,rows=6,cols=8,side=50,gap=1):
    length = find_length(corners[0],corners[1])
    height = find_length(corners[0],corners[2])
    left_up=[0,0]
    left_bottom = [0,length]
    right_up = [height,0]
    # right_bottom = [height,length]
    corners_dst = np.array([left_up, left_bottom, right_up])
    # corners_dst = np.array([left_up, left_bottom, right_up, right_bottom])
    warp_mat = cv2.getAffineTransform(corners.astype(np.float32), corners_dst.astype(np.float32))
    print(warp_mat)
    waped_image = cv2.warpAffine(image,warp_mat,(int(height), int(length)))
    waped_image = cv2.copyMakeBorder(waped_image,20,20,20,20, cv2.BORDER_CONSTANT,value=(100,100,100))
    return waped_image
    
def get_opposite_corners():
    pass

detect_markers()
# generate_markers()