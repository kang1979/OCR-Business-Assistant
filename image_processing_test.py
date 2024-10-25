from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np 
 
def preprocess_image(image):
    """OCR 성능을 높이기 위한 이미지 전처리 함수"""
    
    if len(image.shape) == 3:  # 이미지가 컬러일 경우.
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 흑백 변환
    else:  # 이미지가 이미 그레이스케일인 경우
            gray = image  # 이미지를 그대로 사용

    # 2. 대비 조정git clean -fdx
    gray_image = Image.fromarray(gray)  # OpenCV 이미지를 Pillow 형식으로 변환.
    enhancer = ImageEnhance.Contrast(gray_image)
    enhanced_image = enhancer.enhance(1.5)  # 대비 증가 (값 조정 가능)

    # 3. 해상도 향상
    enhanced_image = enhanced_image.resize((enhanced_image.width * 2, enhanced_image.height * 2), Image.LANCZOS)  # 이미지 확대

    # 4. 선명도 향상
    enhancer = ImageEnhance.Sharpness(enhanced_image)
    enhanced_image = enhancer.enhance(2.0)  # 선명도 증가 (값 조정 가능)

    # Pillow 이미지를 OpenCV 형식으로 변환
    open_cv_image = np.array(enhanced_image)

    # 5. 가우시안 블러 적용
    blurred = cv2.GaussianBlur(open_cv_image, (5, 5), 0)

    # # 6. 이미지 회전 교정
    # coords = np.column_stack(np.where(blurred > 0))
    # angle = cv2.minAreaRect(coords)[-1]
    # if angle < -45:
    #     angle = -(90 + angle)
    # else:
    #     angle = -angle

    # (h, w) = blurred.shape[:2]
    # center = (w // 2, h // 2)
    # M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # rotated = cv2.warpAffine(blurred, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # 7. 이미지 이진화 (적응형 이진화)
    binary = cv2.adaptiveThreshold(blurred, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    return binary

if __name__ == "__main__":
    # 이미지 테스트
    input_image_path = r"media\감사보고서.PNG"  # 입력 이미지 경로
    output_image_path = r"python_test\processed image_test\output_image.png" #출력 이미지 경로

    # 이미지 열기
    try:
        img_array = np.fromfile(input_image_path,np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        # 이미지 전처리 수행
        processed_image = preprocess_image(image)
        # 처리된 이미지 저장
        cv2.imwrite(output_image_pat, processed_image)
        print(f"Processed image saved as {output_image_path}")
    except Exception as e:
        print(f"Error processing image: {str(e)}")
    
# def preprocess_image(image):
#     """이미지 전처리 함수: 대비 조정 및 해상도 향상"""
#     # 1. 흑백 변환
#     image = image.convert("L")  # 그레이스케일로 변환

#     # 2. 대비 증가
#     enhancer = ImageEnhance.Contrast(image)
#     image = enhancer.enhance(1.5)  # 대비 증가 (값 조정 가능)

#     # # 4. 이진화
#     # threshold = 128  # 이진화 임계값 조정
#     # image = image.point(lambda p: 255 if p > threshold else 0)

#     # 5. 해상도 향상
#     image = image.resize((image.width * 7, image.height * 7), Image.LANCZOS)  # 해상도 향상

#     # 6. 선명도 향상
#     enhancer = ImageEnhance.Sharpness(image)
#     image = enhancer.enhance(0)  # 선명도 증가 (값 조정 가능)
#     return image

# if __name__ == "__main__":
#     # 이미지 테스트
#     input_image_path = "media/감사보고서.png"  # 입력 이미지 경로
#     output_image_path = "processed image_test/output_image.png"  # 출력 이미지 경로

#     # 이미지 열기
#     try:
#         image = Image.open(input_image_path)
#         # 이미지 전처리 수행
#         processed_image = preprocess_image(image)
#         # 처리된 이미지 저장
#         processed_image.save(output_image_path)
#         print(f"Processed image saved as {output_image_path}")
#     except Exception as e:
#         print(f"Error processing image: {str(e)}")
