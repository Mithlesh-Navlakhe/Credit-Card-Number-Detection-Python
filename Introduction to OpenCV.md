# OpenCV 介紹

### 讀取、寫入圖片

1. [cv.imread](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_image_display/py_image_display.html)(path, flag)

   + path: 讀取圖像路徑
   + flag: 讀取圖片的方式
     + cv.IMREAD_COLOR
     + cv.IMREAD_GRAYSCALE
     + cv.IMREAD_UNCHANGED
   + 返回: NumPy Array，OpenCV 讀取順序為 BGR

2. [cv.imwrite](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_image_display/py_image_display.html)(filename, img)

   + filename: 存取圖像路徑
   + img: 圖像

3. [cv.imshow](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_image_display/py_image_display.html)(name, img)

   + name: 顯示視窗的名稱

   + img:  圖像

   + 範例: 

     ```python
     cv2.imshow('image',img)
     cv2.waitKey(0)
     cv2.destroyAllWindows()
     ```

### 轉換圖片

1. cv.cvtColor(img, code)
   + img: 圖像
   + code: 轉換代碼
     + cv.COLOR_BGR2GRAY
     + cv.COLOR_BGR2HSV
     + cv.COLOR_BGR2LAB
     + cv.COLOR_BGR2RGB
     + cv.COLOR_LAB2BGR

### 畫圖、插入文字

1. [cv.line](https://docs.opencv.org/master/d6/d6e/group__imgproc__draw.html#ga7078a9fae8c7e7d13d24dac2520ae4a2)(img, start, end, color, thickness=1)
   + img: 背景圖像
   + start: 線條起點
   + end: 線條終點
   + color: 長度為 3 的元組，線條顏色
   + width: 線條寬度
2. [cv.rectangle](https://docs.opencv.org/master/d6/d6e/group__imgproc__draw.html#gac865734d137287c0afb7682ff7b3db23)(img, pt1, pt2, color, width=1)
   + img: 背景圖像
   + pt1: 長方形左上角
   + pt2: 長方形右下角
   + color: 長度為 3 的元組，邊框顏色
   + width: 寬框寬度，若為 1 表示填滿整個長方形
3. [cv.circle](https://docs.opencv.org/master/d6/d6e/group__imgproc__draw.html#gaf10604b069374903dbd0f0488cb43670)(img, center, radius, color, width=1)
   + img: 背景圖像
   + center: 圓心
   + radius: 半徑
   + color: 長度為 3 的元組，邊框顏色
   + width: 寬框寬度，若為 1 表示填滿整個長方形
4. [cv.putText](https://docs.opencv.org/master/d6/d6e/group__imgproc__draw.html#ga5126f47f883d730f633d74f07456c576)(img, str, pt, font, scale, color, thickness=1)
   + img: 背景圖像
   + str: 插入的字串
   + pt: 字串左下角位置
   + font: 字體
     + cv.FONT_HERSHEY_SIMPLEX
     + cv.FONT_HERSHEY_PLAIN
   + scale: 字體大小
   + color: 長度為 3 的元組，字體顏色
   + thickkness: 字體寬度

### 形態學

1. [cv.erode](https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#gaeb1e0c1033e3f6b891a25d0511362aeb)(img,kernel, iterations = 1)

   + img: 圖像
   + kernel: (m, n) 陣列。如果 kernel 範圍內所有像素值都是 1，那麼新的像素值就保持原來的值，否則新的像素值為0
   + iterations: 腐蝕操作的次數

2. [cv.dilate](https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#ga4ff0f3318642c4f469d0e11f242f3b6c)(img,kernel,iterations = 1)

   + img: 圖像
   + kernel: (m, n) 陣列。如果 kernel 範圍內所有像素值都是 0，那麼新的像素值就保持原來的值，否則新的像素值為1 
   + iterations: 膨脹操作的次數

3. 開運算: [cv.morphologyEx](https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#ga67493776e3ad1a3df63883829375201f)(img, cv.MORPH_OPEN, kernel)

   + 開運算: 先使用腐蝕操作在使用膨脹操作，可減少圖片的噪聲
   + img: 圖像
   + kernel

   ![opening.png](https://docs.opencv.org/master/opening.png)

4. 閉運算: [cv.morphologyEx](https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#ga67493776e3ad1a3df63883829375201f)(img, cv.MORPH_CLOSE, kernel)

   + 閉運算: 先使用膨脹操作在使用腐蝕操作，可減少圖片的噪聲
   + img: 圖像
   + kernel 

   ![closing.png](https://docs.opencv.org/master/closing.png)

5. 型態梯度: [cv.morphologyEx](https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#ga67493776e3ad1a3df63883829375201f)(img, cv.MORPH_GRADIENT, kernel)

   + 型態梯度: 膨脹操作減去腐蝕操作
   + img: 圖像
   + kernel

   ![gradient.png](https://docs.opencv.org/master/gradient.png)

6. 頂帽: [cv.morphologyEx](https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#ga67493776e3ad1a3df63883829375201f)(img, cv.MORPH_TOPHAT, kernel)

   + 頂帽: 原始圖像減去開運算
   + img: 圖像
   + kernel

   ![tophat.png](https://docs.opencv.org/master/tophat.png)

7. 黑帽: [cv.morphologyEx](https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#ga67493776e3ad1a3df63883829375201f)(img, cv.MORPH_BLACKHAT, kernel)

   + 黑帽: 閉運算減去原始圖像
   
   + img: 圖像
   
   + kernel
   
     ![blackhat.png](https://docs.opencv.org/master/blackhat.png)









### 模糊化

1.  [cv.blur](https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#ga8c45db9afe636703801b0b2e440fce37)(img, ksize)

   + 對核內像素值做平均
   + img: 圖像
   + ksize: kernel size，元組

2. [cv.GaussianBlur](https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#gaabe8c836e97159a9193fb0b11ac52cf1)(img, ksize, sigmaX, sigmaY=0)

   + 使用高斯核對像素值做平均，使用此方法移除高斯噪聲很有效
   + img: 圖像
   + ksize: kernel size，元組
   + (sigmaX, sigmaY):
     + 為 x 與 y 方向的變異數
     + 若 sigmaX = 0, 則 y 方向的變異數等於 x 方向變異數
     + 若 sigmaX = sigmaY = 0, 則 x 方向變異數 ksize.width, y 方向變異數為 ksize.height

3.  [cv.medianBlur](https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#ga564869aa33e58769b4469101aac458f9)(img, ksize)

   + 對核中的數取中位數，使用此方法移除 salt-and-pepper noise 很有效

   + ksize: kernel size，核的長寬大小，需為基數，使用的核為正方形

     <img src="https://blog.photopea.com/wp-content/uploads/2016/09/head.jpg" alt="img" style="zoom: 50%;" />

4. [cv.bilateralFilter](https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#ga9d7064d478c95d60003cf839430737ed)(img, d, sigmaColor, sigmaSpace)
   + img: 圖像
   + d
   + sigmaColor
   + sigmaSpace

### 梯度

1. 概述: 輪廓通常發生在梯度值較大的地方，因此，我們可以透過計算梯度並找出梯度極值所在點來判斷物體的輪廓。[參考資料](https://docs.opencv.org/3.4/d2/d2c/tutorial_sobel_derivatives.html)

2. [cv.Sobel](https://docs.opencv.org/master/d4/d86/group__imgproc__filter.html#gacea54f142e81b6758cb6f375ce782c8d)(img, ddepth, dx, dy, ksize=3)

   + 當 ksize = 3 時，Sobel 函數分別使用以下方式求出 x, y 方向的梯度
     $$
     G_{x} = 
     \begin{bmatrix} -1 & 0 & +1 \\ -2 & 0 & +2 \\ -1 & 0 & +1 \end{bmatrix}* I, \quad \quad
     G_{y} = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ +1 & +2 & +1 \end{bmatrix}* I.
     $$
     我們通常使用
     $$
     G = \sqrt{ G_{x}^{2} + G_{y}^{2} } \quad 或 \quad G = |G_{x}| + |G_{y}|
     $$
     求梯度值

   + img: 圖像

   + ddepth: 圖像深度，指存儲每個像素值所用的位數，超出範圍則會被截斷

     + CV_16S: 16 位無符號數
     + CV_16U: 16 位有符號數
     + CV_32F: 32 位浮點數
     + CV_64F: 64 位浮點數

   + dx: x 方向求導的階數

   + dy: y 方向求導的階數

   + ksize: kernel size

3. [cv.Scharr](https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#gaa13106761eedf14798f37aa2d60404c9)(img, ddepth, dx, dy, ksize=3)

   + 原理同 sobel 函數，當 ksize = 3 時，Scharr 函數分別使用以下方式求出 x, y 方向的梯度
     $$
     G_{x} = \begin{bmatrix} -3 & 0 & +3 \\ -10 & 0 & +10 \\ -3 & 0 & +3 \end{bmatrix}*I, \quad
     G_{y} = \begin{bmatrix} -3 & -10 & -3 \\ 0 & 0 & 0 \\ +3 & +10 & +3 \end{bmatrix}*I.
     $$

4. [cv.Laplacian](https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#gad78703e4c8fe703d479c1860d76429e6)(img, ddpeth, ksize=1)

   + img
   + ddepth
   + ksize

### Canny 演算法

1. [cv.Canny]((https://docs.opencv.org/master/da/d22/tutorial_py_canny.html))(image, threshold1, threshold2)

   + Canny 邊緣檢測步驟:
     + 將圖片轉換為灰值，並使用高斯過濾器去除雜訊
     + 取得每個像素的**梯度值**與**梯度方向**
     + 使用非極大值抑制尋找邊界 (Non-maximum suppression):  把每個像素和梯度方向的鄰居比較梯度值，如果不是最大的，就去除該像素
     + 根據 threshold1, threshold2 選取 strong edge, weak edge
     + 最後選取 strong edge 與 與 strong edge 相連的 weak edge

   + img: 圖像
   + (threshold1, thresold2):
     +  大於 threshold1: strong edge
     + 小於 threshold2: 去除
     + 介於 threshold1 與 threshold2 之間: weak edge 



### 閾值

1. [cv.threshold](https://docs.opencv.org/master/d7/d1b/group__imgproc__misc.html#gae8a4a146d1ca78c626a53577199e9c57)(img, threshold, maxval, cv.THRESH_BINARY)

   + 輸出函數
     $$
     dst(x,y)=
     \left\{  
                  \begin{array}{**lr**}
                  \rm maxval, & src(x,y) > \rm threshold\\  
                  0,\quad & \rm otherwise \\
                  \end{array}  
     \right.
     $$
+ img: 圖像，需為灰度圖
   + threshold: 設定閾值 
   
   + maxval: 超過閾值的像素設定的亮度值
   
     ![Threshold_Tutorial_Theory_Binary.png](https://docs.opencv.org/3.4/Threshold_Tutorial_Theory_Binary.png)
   
     2.  [cv.threshold](https://docs.opencv.org/master/d7/d1b/group__imgproc__misc.html#gae8a4a146d1ca78c626a53577199e9c57)(img, threshold, 

2. [cv.threshold](https://docs.opencv.org/master/d7/d1b/group__imgproc__misc.html#gae8a4a146d1ca78c626a53577199e9c57)(img, threshold, maxval, cv.THRESH_BINARY_INV)

   + 輸出函數
     $$
     dst(x,y)=
     \left\{  
                  \begin{array}{**lr**}
                  0, & src(x,y) > \rm threshold\\  
                  \rm maxval,\quad & \rm otherwise \\
                  \end{array}  
     \right.
     $$

   + img: 圖像，需為灰度圖

   + threshold: 設定閾值

   + maxval: 低於閾值像素設定的亮度值

     ![Threshold_Tutorial_Theory_Binary_Inverted.png](https://docs.opencv.org/3.4/Threshold_Tutorial_Theory_Binary_Inverted.png)

   

3. [cv.threshold](https://docs.opencv.org/master/d7/d1b/group__imgproc__misc.html#gae8a4a146d1ca78c626a53577199e9c57)(img, threshold, maxval, cv.THRESH_TRUNC)

   + 輸出函數: 將大於門檻值的亮度值設為 threshold，小於門檻值的值保持不變
     $$
     dst(x,y)=\left\{               
     \begin{array}{**lr**}             
     \rm threshold, & src(x,y) > \rm threshold\\               
     \rm src(x,y),\quad & \rm otherwise \\             
     \end{array}  \right.
     $$
+ img: 圖像，需為灰度圖
  
+ threshold: 設定閾值
  
+ maxval: 無功能
  
  ![Threshold_Tutorial_Theory_Truncate.png](https://docs.opencv.org/3.4/Threshold_Tutorial_Theory_Truncate.png)



4.  [cv.threshold](https://docs.opencv.org/master/d7/d1b/group__imgproc__misc.html#gae8a4a146d1ca78c626a53577199e9c57)(img, threshold, maxval, cv.THRESH_TOZERO)

   + 輸出函數: 將小於門檻值的灰階值設為0，大於門檻值的值保持不變
     $$
     dst(x,y)=\left\{               
     \begin{array}{**lr**}             
     \rm src(x,y), & src(x,y) > \rm threshold\\               
     \rm 0,\quad & \rm otherwise \\             
     \end{array}  \right.
     $$

   + img: 圖像，需為灰度圖

   + threshold: 設定閾值

   + maxval: 無功能

     ![Threshold_Tutorial_Theory_Zero.png](https://docs.opencv.org/3.4/Threshold_Tutorial_Theory_Zero.png)

5.  [cv.adaptiveThreshold](https://docs.opencv.org/master/d7/d1b/group__imgproc__misc.html#ga72b913f352e4a1b1b397736707afcde3)(img, maxval, adaptiveMethod, type, blockSize, C)
   + 操作: 根據圖片分塊設定閾值，並使用二值化函數操作	
   
   + img: 圖像，需為灰度圖
   
   + maxval
   
   + adaptiveMethod: 
     + cv.ADAPTIVE_THRESH_MEAN_C: 使用分塊像素之平均作為閾值 
     + cv.ADAPTIVE_THRESH_GAUSSIAN_C: 使用分塊像素之高斯平均作為閾值 
     
   + type:
     + cv.THRESH_BINARY
     + cv.THRESH_BINARY_INV
     + cv.THRESH_TRUNC
     + cv.THRESH_TOZERO
     
   + blockSize: 圖像分塊的大小
   
   + C: 常數，用來減去分塊中的平均值作為分塊的閾值
   
     ![ada_threshold.jpg](https://docs.opencv.org/master/ada_threshold.jpg)



6. Otsu’s Binarization: [cv.threshold](https://docs.opencv.org/master/d7/d1b/group__imgproc__misc.html#gae8a4a146d1ca78c626a53577199e9c57)(img, threshold, maxval, cv.THRESH_BINARY + cv.THRESH_OTSU)
   + 操作: Otsu’s Binarization 會自動找出圖片的閾值，並根據該閾值使用二值化操作。圖像亮度為雙峰分佈時，閾值為雙峰的位置的中點；若圖像亮度分佈不為二值，此方法效果並不是很好
   + img: 圖像，需為灰度圖
   + threshold
   + maxval

### Contour 操作

1. 概述: 

   + Contour 為一條畫出邊界的連續線，且該線上顏色都一樣
   + 使用二值圖像進行 Contour 操作效果會更好，因此，在之前最好先使用 cv.threshold 或 cv.Canny 操作
   + 進行二值化處理時，請將要畫出邊界的物體設為白色，背景為黑色 

2.  尋找輪廓: [cv.findContours](https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html#gae4156f04053c44f886e387cff0ef6e08)(img, mode, method)

   + img: 圖像
   + mode: 輪廓檢索模式
     + cv.RETR_EXTERNAL: 只檢測外輪廓
     + cv.RETR_LIST: 輪廓不建立等級關係
     + cv.RETR_TREE: 建立等級結構的輪廓
   + method: 輪廓近似方法
     +  cv.CHAIN_APPROX_NON: 儲存所有輪廓點
     +  cv.CHAIN_APPROX_SIMPLE: 壓縮水平、垂直或對角的線段，只記錄這些線段的起點與終點
   + 返回: im2, contours, hierarchy

   + 範例:

     ```python
     import numpy as np
     import cv2 as cv
     im = cv.imread('test.jpg')
     imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
     ret, thresh = cv.threshold(imgray, 127, 255, 0)
     im2, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
     ```

     

3. 畫出輪廓: [cv.drawContours](https://docs.opencv.org/master/d6/d6e/group__imgproc__draw.html#ga746c0625f1781f1ffc9056259103edbc)(img, contours, contourIdx, color, thickness)

   + img: 背景圖像
   + contours: 輪廓，通常使用 findContours 中返回的 contours 做為參數
   + contoursIdx:  要畫第幾個輪廓，-1 表示顯示所有輪廓
   + color 顏色，長度為 3 的元組
   + thickness: 輪廓粗度

   
   
   ![none.jpg](https://docs.opencv.org/3.4/none.jpg)
   
   

4. 畫出 BBox: [cv.boundingRect](https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga103fcbda2f540f3ef1c042d6a9b35ac7)(contour)

   + 畫出包含 contour 的最小長方形，不考慮旋轉過後的長方形

   + 範例

     ```python
     x,y,w,h = cv.boundingRect(cnt)
     cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
     ```

     

   