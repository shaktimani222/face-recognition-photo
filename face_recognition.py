# ==================================================
# FIXED FACE RECOGNITION - STRICT MATCH
# ==================================================

# pip install deepface opencv-python matplotlib

from deepface import DeepFace
import cv2, os
import matplotlib.pyplot as plt

known_img = "passport shaktimani2.jpg"
test_img  = "IMG_4182.jpg"
person_name = "Shaktimani"

# database
os.makedirs("/content/database", exist_ok=True)
cv2.imwrite("/content/database/Shaktimani.jpg", cv2.imread(known_img))

img = cv2.imread(test_img)

faces = DeepFace.extract_faces(
    img_path=test_img,
    detector_backend="retinaface",
    enforce_detection=True
)

for face in faces:

    area = face["facial_area"]
    x,y,w,h = area["x"], area["y"], area["w"], area["h"]

    crop = img[y:y+h, x:x+w]
    cv2.imwrite("/content/temp.jpg", crop)

    label = "Unknown"
    color = (0,0,255)

    try:
        result = DeepFace.find(
            img_path="/content/temp.jpg",
            db_path="/content/database",
            model_name="ArcFace",
            detector_backend="retinaface",
            silent=True
        )

        if len(result[0]) > 0:

            distance = result[0].iloc[0]["distance"]

            # STRICT threshold
            if distance < 0.40:
                label = person_name
                color = (0,255,0)

    except:
        pass

    cv2.rectangle(img,(x,y),(x+w,y+h),color,3)
    cv2.rectangle(img,(x,y-35),(x+w,y),color,-1)
    cv2.putText(img,label,(x+5,y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,(255,255,255),2)

cv2.imwrite("/content/final_fixed.jpg",img)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(14,8))
plt.imshow(img_rgb)
plt.axis("off")
plt.show()