import cv2
import numpy as np

image1 = cv2.imread('image_name.jpg')
image2 = cv2.imread('image_name1.jpg')

if image1 is None:
    print("Error: Could not load image 'image_name.jpg'. Check the file path.")
if image2 is None:
    print("Error: Could not load image 'image_name1.jpg'. Check the file path.")

gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

def detect_harris_corners(image, original_image):
    gray = np.float32(image)
    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    dst = cv2.dilate(dst, None)

    
    mask = dst > 0.01 * dst.max()
    
    output_image = original_image.copy()
    output_image[mask] = [0, 0, 255]
    
    return output_image, np.argwhere(mask)

corners1_img, corners1 = detect_harris_corners(gray1.copy(), image1)
corners2_img, corners2 = detect_harris_corners(gray2.copy(), image2)

sift = cv2.SIFT_create()

keypoints1 = [cv2.KeyPoint(float(pt[1]), float(pt[0]), 1) for pt in corners1]
keypoints2 = [cv2.KeyPoint(float(pt[1]), float(pt[0]), 1) for pt in corners2]

keypoints1, descriptors1 = sift.compute(gray1, keypoints1)
keypoints2, descriptors2 = sift.compute(gray2, keypoints2)

keypoints1_img = cv2.drawKeypoints(image1, keypoints1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
keypoints2_img = cv2.drawKeypoints(image2, keypoints2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

def match_features(desc1, desc2):
    matches = []
    for i in range(desc1.shape[0]):
        dists = np.linalg.norm(desc2 - desc1[i], axis=1)
        sorted_idx = np.argsort(dists)
        best_match = sorted_idx[0]
        second_best_match = sorted_idx[1]
        ratio = dists[best_match] / dists[second_best_match]
        matches.append((i, best_match, ratio))
    return matches

matches = match_features(descriptors1, descriptors2)

good_matches = [m for m in matches if m[2] < 0.75]

img_matches = cv2.drawMatches(image1, keypoints1, image2, keypoints2,
                              [cv2.DMatch(_queryIdx=m[0], _trainIdx=m[1], _imgIdx=0, _distance=0) for m in good_matches],
                              None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

def display_image(img, title):
    cv2.imshow(title, img)
    cv2.waitKey(0)  
    cv2.destroyAllWindows()


display_image(corners1_img, 'Harris Corners Image 1')
display_image(corners2_img, 'Harris Corners Image 2')
display_image(keypoints1_img, 'SIFT Keypoints Image 1')
display_image(keypoints2_img, 'SIFT Keypoints Image 2')
display_image(img_matches, 'Feature Matches')


cv2.imwrite('corners_image1.png', corners1_img)
cv2.imwrite('corners_image2.png', corners2_img)
cv2.imwrite('sift_keypoints_image1.png', keypoints1_img)
cv2.imwrite('sift_keypoints_image2.png', keypoints2_img)
cv2.imwrite('feature_matches.png', img_matches)

