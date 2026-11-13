# REVIEW — ex08, leaf classifier in the browser

1. It runs on your phone
- image0.png 

2. The same numbers as the notebook
- The four output values match to three decimal places: 0.999 / 0.001 / 0.000 / 0.000. The largest difference is extremely small (0.000000064). The self-test passed, and the Python reference gives almost identical results.
-One limitation: check_image.jpg is already 96×96 pixels, so this test checks image normalisation, colour channel order and probability calculations, but does not test image cropping. Processing a live camera image produces a slightly different result, but not enough to change the prediction.

3. The preprocesssing matches your model
-The actual image-processing code on the page  correctly crops and resizes the image to 96×96 pixels. The colour channels are also in the correct order.
-Repeating the same image processing in Python produces almost identical results. Using the wrong colour channel order creates a small difference, while using the wrong image format causes the model to reject the input.
-A detailed, line-by-line comparison is included in the document (index.html, lines 329–377).
-The original ex08 notebook was not available in the folder, so the comparison was made using its exported check_expected.json file.

4. Nothing leaves the phone
-The page connects to only two websites: 257061.github.io and cdn.jsdelivr.net. There were 14 network requests, all GET requests, with no data uploaded. During approximately 120 live camera analyses, no network requests were made.
-The code also does not use XHR, WebSockets, tracking beacons or browser storage.
-One important clarification: although no images are uploaded, the two websites can still see the visitor's IP address, just like with any normal website.


