import numpy as np
import cv2
from matplotlib import pyplot as plt
import pickle
import tensorflow as tf
import matplotlib.colors as mcolors

# Load the trained KNN model
classifier = pickle.load(open('knn.sav', 'rb'))

#Modify the filepath according to user.
filepath = 'E:\\DIP\\Project\\Automatic Sudoku Solver\\Images\\su3.jpg'
# Read the image
Img = cv2.imread(filepath)
Img1 = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)

# Convert the image to grayscale
Img_gray = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)
plt.subplot(121), plt.imshow(Img), plt.title('Original Image')
plt.subplot(122), plt.imshow(Img_gray, cmap='gray'), plt.title('Grayscale Image')
plt.show()

#Gaussian Blurring the image
gauss = cv2.GaussianBlur(Img_gray,(11,11),0)
plt.imshow(gauss,cmap ='gray'), plt.title('Gaussian Blurred Image')
plt.show()

#Using adaptive thresholding to account for uneven brightness
gauss_T = cv2.adaptiveThreshold(gauss, 255, cv2.ADAPTIVE_THRESH_MEAN_C | cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)
plt.imshow(gauss_T,cmap ='gray'), plt.title('Adaptive Thresh')
plt.show()

#Inverting the image
Img_inv = cv2.bitwise_not(gauss_T)
plt.imshow(Img_inv,cmap ='gray'), plt.title('Inverted')
plt.show()

#Dilating the image
kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
Img_dil = cv2.dilate(Img_inv, kernel)
plt.imshow(Img_dil,cmap ='gray'), plt.title('Dilated')
plt.show()

#Flood filling to find the biggest blob in the image
Img_fill = Img_dil.copy()
maxi = -1
maxpt = None
value = 10
height, width = np.shape(Img_fill)
for y in range(height):
    row = Img_dil[y]
    for x in range(width):
        if row[x] >= 128:
            area = cv2.floodFill(Img_fill, None, (x, y), 64)[0]
            if area > maxi:
                maxpt = (x, y)
                maxi = area

# Floodfill the biggest blob with white (Our sudoku board's outer grid)
cv2.floodFill(Img_fill, None, maxpt, (255, 255, 255))

# Floodfill the other blobs with black
for y in range(height):
    row = Img_fill[y]
    for x in range(width):
        if row[x] == 64 and x != maxpt[0] and y != maxpt[1]:
            cv2.floodFill(Img_fill, None, (x, y), 0)

plt.imshow(Img_fill,cmap ='gray'), plt.title('Floodfilled Image')
plt.show()

#Eroding the image
kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
Img_erode = cv2.erode(Img_fill, kernel)
plt.imshow(Img_erode,cmap ='gray'), plt.title('Eroded Image')
plt.show()

lines = cv2.HoughLines(Img_erode, 1, np.pi / 180, 200)

#This function takes a line in it's normal form and draws it on an image
def drawLine(line, img):
    img_with_lines = np.copy(img)
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(img_with_lines, (x1, y1), (x2, y2), 255, 2)
    return img_with_lines


# Draw and display all the lines
tmpimg = np.copy(Img_erode)
for i in range(len(lines)):
    tmpimp = drawLine(lines[i], tmpimg)
Img_line = tmpimg
plt.imshow(Img_line,cmap ='gray'), plt.title('Hough Lines')
plt.show()


#Mergelines
def mergeLines(lines, img):
    height, width = np.shape(img)
    for current in lines:
        if current[0][0] is None and current[0][1] is None:
            continue
        p1 = current[0][0]
        theta1 = current[0][1]
        pt1current = [None, None]
        pt2current = [None, None]
        # If the line is almost horizontal
        if theta1 > np.pi * 45 / 180 and theta1 < np.pi * 135 / 180:
            pt1current[0] = 0
            pt1current[1] = p1 / np.sin(theta1)
            pt2current[0] = width
            pt2current[1] = -pt2current[0] / np.tan(theta1) + p1 / np.sin(theta1)
        # If the line is almost vertical
        else:
            pt1current[1] = 0
            pt1current[0] = p1 / np.cos(theta1)
            pt2current[1] = height
            pt2current[0] = -pt2current[1] * np.tan(theta1) + p1 / np.cos(theta1)
        # Now to fuse lines
        for pos in lines:
            if pos[0].all() == current[0].all():
                continue
            if abs(pos[0][0] - current[0][0]) < 20 and abs(pos[0][1] - current[0][1]) < np.pi * 10 / 180:
                p = pos[0][0]
                theta = pos[0][1]
                pt1 = [None, None]
                pt2 = [None, None]
                # If the line is almost horizontal
                if theta > np.pi * 45 / 180 and theta < np.pi * 135 / 180:
                    pt1[0] = 0
                    pt1[1] = p / np.sin(theta)
                    pt2[0] = width
                    pt2[1] = -pt2[0] / np.tan(theta) + p / np.sin(theta)
                # If the line is almost vertical
                else:
                    pt1[1] = 0
                    pt1[0] = p / np.cos(theta)
                    pt2[1] = height
                    pt2[0] = -pt2[1] * np.tan(theta) + p / np.cos(theta)
                # If the endpoints are close to each other, merge the lines
                if (pt1[0] - pt1current[0]) ** 2 + (pt1[1] - pt1current[1]) ** 2 < 64 ** 2 and (
                        pt2[0] - pt2current[0]) ** 2 + (pt2[1] - pt2current[1]) ** 2 < 64 ** 2:
                    current[0][0] = (current[0][0] + pos[0][0]) / 2
                    current[0][1] = (current[0][1] + pos[0][1]) / 2
                    pos[0][0] = None
                    pos[0][1] = None
    # Now to remove the "None" Lines
    lines = list(filter(lambda a: a[0][0] is not None and a[0][1] is not None, lines))
    return lines

# Call the Merge Lines function and store the fused lines
lines = mergeLines(lines, Img_erode)

topedge = [[1000, 1000]]
bottomedge = [[-1000, -1000]]
leftedge = [[1000, 1000]]
leftxintercept = 100000
rightedge = [[-1000, -1000]]
rightxintercept = 0

for i in range(len(lines)):
    current = lines[i][0]
    p = current[0]
    theta = current[1]
    xIntercept = p / np.cos(theta)

    # If the line is nearly vertical
    if theta > np.pi * 80 / 180 and theta < np.pi * 100 / 180:
        if p < topedge[0][0]:
            topedge[0] = current[:]
        if p > bottomedge[0][0]:
            bottomedge[0] = current[:]

    # If the line is nearly horizontal
    if theta < np.pi * 10 / 180 or theta > np.pi * 170 / 180:
        if xIntercept > rightxintercept:
            rightedge[0] = current[:]
            rightxintercept = xIntercept
        elif xIntercept <= leftxintercept:
            leftedge[0] = current[:]
            leftxintercept = xIntercept

# Drawing the lines
tmpimg = np.zeros_like(Img_erode)  # Create a black image with the same size as Img_erode
tmppp = np.copy(Img1)

tmpimg = np.copy(Img_erode)
tmppp = np.copy(Img1)
tmppp = drawLine(leftedge, tmppp)
tmppp = drawLine(rightedge, tmppp)
tmppp = drawLine(topedge, tmppp)
tmppp = drawLine(bottomedge, tmppp)

tmpimg = drawLine(leftedge, tmpimg)
tmpimg = drawLine(rightedge, tmpimg)
tmpimg = drawLine(topedge, tmpimg)
tmpimg = drawLine(bottomedge, tmpimg)

plt.imshow(tmpimg,cmap ='gray'), plt.title('Hough Lines')
plt.show()

plt.imshow(tmppp,cmap ='gray'), plt.title('Hough Lines')
plt.show()

leftedge = leftedge[0]
rightedge = rightedge[0]
bottomedge = bottomedge[0]
topedge = topedge[0]

# Calculating two points that lie on each of the four lines
left1 = [None, None]
left2 = [None, None]
right1 = [None, None]
right2 = [None, None]
top1 = [None, None]
top2 = [None, None]
bottom1 = [None, None]
bottom2 = [None, None]

if leftedge[1] != 0:
    left1[0] = 0
    left1[1] = leftedge[0] / np.sin(leftedge[1])
    left2[0] = width
    left2[1] = -left2[0] / np.tan(leftedge[1]) + left1[1]
else:
    left1[1] = 0
    left1[0] = leftedge[0] / np.cos(leftedge[1])
    left2[1] = height
    left2[0] = left1[0] - height * np.tan(leftedge[1])

if rightedge[1] != 0:
    right1[0] = 0
    right1[1] = rightedge[0] / np.sin(rightedge[1])
    right2[0] = width
    right2[1] = -right2[0] / np.tan(rightedge[1]) + right1[1]
else:
    right1[1] = 0
    right1[0] = rightedge[0] / np.cos(rightedge[1])
    right2[1] = height
    right2[0] = right1[0] - height * np.tan(rightedge[1])

bottom1[0] = 0
bottom1[1] = bottomedge[0] / np.sin(bottomedge[1])

bottom2[0] = width
bottom2[1] = -bottom2[0] / np.tan(bottomedge[1]) + bottom1[1]

top1[0] = 0
top1[1] = topedge[0] / np.sin(topedge[1])
top2[0] = width
top2[1] = -top2[0] / np.tan(topedge[1]) + top1[1]

# Next, we find the intersection of these four lines

leftA = left2[1] - left1[1]
leftB = left1[0] - left2[0]
leftC = leftA * left1[0] + leftB * left1[1]

rightA = right2[1] - right1[1]
rightB = right1[0] - right2[0]
rightC = rightA * right1[0] + rightB * right1[1]

topA = top2[1] - top1[1]
topB = top1[0] - top2[0]
topC = topA * top1[0] + topB * top1[1]

bottomA = bottom2[1] - bottom1[1]
bottomB = bottom1[0] - bottom2[0]
bottomC = bottomA * bottom1[0] + bottomB * bottom1[1]

# Intersection of left and top

detTopLeft = leftA * topB - leftB * topA

ptTopLeft = ((topB * leftC - leftB * topC) / detTopLeft, (leftA * topC - topA * leftC) / detTopLeft)

# Intersection of top and right

detTopRight = rightA * topB - rightB * topA

ptTopRight = ((topB * rightC - rightB * topC) / detTopRight, (rightA * topC - topA * rightC) / detTopRight)

# Intersection of right and bottom

detBottomRight = rightA * bottomB - rightB * bottomA

ptBottomRight = ((bottomB * rightC - rightB * bottomC) / detBottomRight, (rightA * bottomC - bottomA * rightC) / detBottomRight)

# Intersection of bottom and left

detBottomLeft = leftA * bottomB - leftB * bottomA

ptBottomLeft = ((bottomB * leftC - leftB * bottomC) / detBottomLeft,
                       (leftA * bottomC - bottomA * leftC) / detBottomLeft)

# Plotting the found extreme points
cv2.circle(tmppp, (int(ptTopLeft[0]), int(ptTopLeft[1])), 5, 0, -1)
cv2.circle(tmppp, (int(ptTopRight[0]), int(ptTopRight[1])), 5, 0, -1)
cv2.circle(tmppp, (int(ptBottomLeft[0]), int(ptBottomLeft[1])), 5, 0, -1)
cv2.circle(tmppp, (int(ptBottomRight[0]), int(ptBottomRight[1])), 5, 0, -1)

plt.imshow(tmppp, cmap= 'gray'), plt.title('Edge Detected')
plt.show()

# Finding the maximum length side
leftedgelensq = (ptBottomLeft[0] - ptTopLeft[0])**2 + (ptBottomLeft[1] - ptTopLeft[1])**2
rightedgelensq = (ptBottomRight[0] - ptTopRight[0])**2 + (ptBottomRight[1] - ptTopRight[1])**2
topedgelensq = (ptTopRight[0] - ptTopLeft[0])**2 + (ptTopLeft[1] - ptTopRight[1])**2
bottomedgelensq = (ptBottomRight[0] - ptBottomLeft[0])**2 + (ptBottomLeft[1] - ptBottomRight[1])**2
maxlength = int(max(leftedgelensq, rightedgelensq, bottomedgelensq, topedgelensq)**0.5)

# Correcting the skewed perspective
src = [(0, 0)] * 4
dst = [(0, 0)] * 4
src[0] = ptTopLeft[:]
dst[0] = (0, 0)
src[1] = ptTopRight[:]
dst[1] = (maxlength - 1, 0)
src[2] = ptBottomRight[:]
dst[2] = (maxlength - 1, maxlength - 1)
src[3] = ptBottomLeft[:]
dst[3] = (0, maxlength - 1)
src = np.array(src).astype(np.float32)
dst = np.array(dst).astype(np.float32)
extractedgrid = cv2.warpPerspective(Img1, cv2.getPerspectiveTransform(src, dst), (maxlength, maxlength))
# Resizing the grid to a 252X252 size because MNIST has 28X28 images
extractedgrid = cv2.resize(extractedgrid, (252, 252))

plt.imshow(extractedgrid, cmap = 'gray'), plt.title('Extracted Grid')
plt.show()

def create_image_grid(extractedgrid):
    grid = np.copy(extractedgrid)
    edge = np.shape(grid)[0]
    celledge = edge // 9

    # Adaptive thresholding the cropped grid and inverting it
    grid = cv2.bitwise_not(cv2.adaptiveThreshold(grid, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 55 , 1))
    # Creating a vector of size 81 of all the cell images
    tempgrid = []
    for i in range(celledge, edge + 1, celledge):
        for j in range(celledge, edge + 1, celledge):
            rows = grid[i - celledge:i]
            tempgrid.append([rows[k][j - celledge:j] for k in range(len(rows))])

    # Creating the 9X9 grid of images
    finalgrid = []
    for i in range(0, len(tempgrid) - 8, 9):
        finalgrid.append(tempgrid[i:i + 9])

    # Converting all the cell images to np.array and displaying them
    for i in range(9):
        for j in range(9):
            finalgrid[i][j] = np.array(finalgrid[i][j])
            plt.subplot(9, 9, i * 9 + j + 1)
            plt.imshow(finalgrid[i][j], cmap='gray')
            plt.axis('off')

    plt.show()
    

    return finalgrid

Img_grid = create_image_grid(extractedgrid)
print(np.size(Img_grid))

def construct_grid(cellarray):
    finalgrid1 = [[0 for _ in range(9)] for _ in range(9)]
    threshold = 5 * 255

    for i in range(9):
        for j in range(9):
            tmp = np.copy(cellarray[i][j])
            tmp = preprocess_image(tmp)
            cv2.imwrite(f"CleanedBoardCells/cell{i}{j}.jpg", tmp)
            finsum = np.sum(tmp)
            if finsum < threshold:
                finalgrid1[i][j] = 0
                continue
            pred = make_prediction(tmp)
            finalgrid1[i][j] = int(pred)
    return finalgrid1

def make_prediction(cleanedimg):
    try:
        print("Processing image...")
        if MODEL_TYPE == "CNN":
            lis = [cleanedimg]
            lis = np.reshape(lis, (1, 28, 28, 1))
            idx = None
            pred = CNN_MODEL.predict(lis)
            for i in range(len(pred[0])):
                if pred[0][i] > 0:
                    idx = i
                    break
            return idx
        elif MODEL_TYPE == "KNN":
            cleanedimg = cleanedimg.reshape(1, -1)
            prediction = KNN_MODEL.predict(cleanedimg)[0]
            return prediction
    except Exception as e:
        print(f"Error: {e}")
        return None
def preprocess_image(img):
    rows = np.shape(img)[0]

    for i in range(rows):
        cv2.floodFill(img, None, (0, i), 0)
        cv2.floodFill(img, None, (i, 0), 0)
        cv2.floodFill(img, None, (rows - 1, i), 0)
        cv2.floodFill(img, None, (i, rows - 1), 0)
        cv2.floodFill(img, None, (1, i), 1)
        cv2.floodFill(img, None, (i, 1), 1)
        cv2.floodFill(img, None, (rows - 2, i), 1)
        cv2.floodFill(img, None, (i, rows - 2), 1)

    rowtop, rowbottom, colleft, colright = None, None, None, None
    thresholdBottom = thresholdTop = thresholdLeft = thresholdRight = 50
    center = rows // 2

    for i in range(center, rows):
        if rowbottom is None:
            temp = img[i]
            if np.sum(temp) < thresholdBottom or i == rows - 1:
                rowbottom = i
        if rowtop is None:
            temp = img[rows - i - 1]
            if np.sum(temp) < thresholdTop or i == rows - 1:
                rowtop = rows - i - 1
        if colright is None:
            temp = img[:, i]
            if np.sum(temp) < thresholdRight or i == rows - 1:
                colright = i
        if colleft is None:
            temp = img[:, rows - i - 1]
            if np.sum(temp) < thresholdLeft or i == rows - 1:
                colleft = rows - i - 1

    newimg = np.zeros(np.shape(img))
    startatX = (rows + colleft - colright) // 2
    startatY = (rows - rowbottom + rowtop) // 2

    for y in range(startatY, (rows + rowbottom - rowtop) // 2):
        for x in range(startatX, (rows - colleft + colright) // 2):
            newimg[y, x] = img[rowtop + y - startatY, colleft + x - startatX]

    return newimg



def display_sudoku_grid(final_grid):
    """
    Display the final Sudoku grid using matplotlib.
    """
    plt.figure(figsize=(6, 6))
    cmap = plt.get_cmap('viridis', np.max(final_grid) - np.min(final_grid) + 1)
    norm = mcolors.BoundaryNorm(np.arange(np.min(final_grid) - 0.5, np.max(final_grid) + 0.5, 1), cmap.N)
    
    plt.imshow(final_grid, cmap=cmap, norm=norm, interpolation='none', aspect='auto')

    for i in range(len(final_grid)):
        for j in range(len(final_grid[i])):
            value = final_grid[i, j]
            plt.annotate(str(value) if value != 0 else '', (j, i), color='black', ha='center', va='center', fontsize=12)

    plt.xticks([]), plt.yticks([])
    plt.show()


# Assuming these are global variables in your original code
MODEL_TYPE = "KNN"
CNN_MODEL = tf.keras.models.load_model("cnn.hdf5") if MODEL_TYPE == "CNN" else None
KNN_MODEL = pickle.load(open("knn.sav", 'rb')) if MODEL_TYPE == "KNN" else None

Final_Grid = construct_grid(Img_grid)

# Convert the final Sudoku grid to a numpy array for visualization
final_grid_array = np.array(Final_Grid)
print(final_grid_array)
# Display the final Sudoku grid
display_sudoku_grid(final_grid_array)
def display_sudoku_with_overlay(original_image, final_grid):
    """
    Display the original image and the final Sudoku grid side by side.
    """
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Image')
    
    plt.subplot(1,2,2)
    cmap = plt.get_cmap('viridis', np.max(final_grid) - np.min(final_grid) + 1)
    norm = mcolors.BoundaryNorm(np.arange(np.min(final_grid) - 0.5, np.max(final_grid) + 0.5, 1), cmap.N)
    
    plt.imshow(final_grid, cmap=cmap, norm=norm, interpolation='none', aspect='auto')

    for i in range(len(final_grid)):
        for j in range(len(final_grid[i])):
            value = final_grid[i, j]
            plt.annotate(str(value) if value != 0 else '', (j, i), color='black', ha='center', va='center', fontsize=12)

    plt.xticks([]), plt.yticks([])
    
    plt.show()

def get_user_corrections(final_grid):
    """
    Get user corrections for the final Sudoku grid.
    """
    corrections = []
    
    while True:
        user_input = input("Do you want to make corrections? (y/n): ").lower()
        
        if user_input == 'n':
            break
        elif user_input == 'y':
            try:
                row = int(input("Enter row number (1-9): ")) - 1
                col = int(input("Enter column number (1-9): ")) - 1
                value = int(input("Enter corrected value (1-9): "))
                
                if 0 <= row < 9 and 0 <= col < 9 and 1 <= value <= 9:
                    corrections.append((row, col, value))
                    print(f"Correction recorded: ({row + 1}, {col + 1}) -> {value}")
                else:
                    print("Invalid input. Please enter valid values.")
            except ValueError:
                print("Invalid input. Please enter numeric values.")
        else:
            print("Invalid input. Please enter 'y' or 'n'.")
    
    return corrections

def apply_user_corrections(final_grid, corrections):
    """
    Apply user corrections to the final Sudoku grid.
    """
    for row, col, value in corrections:
        final_grid[row][col] = value


# Display the original image and final Sudoku grid
display_sudoku_with_overlay(Img, final_grid_array)

# Get user corrections
corrections = get_user_corrections(final_grid_array)

# Apply user corrections
apply_user_corrections(final_grid_array, corrections)

# Display the updated final Sudoku grid
display_sudoku_with_overlay(Img, final_grid_array)

#Display corrected sudoku

print(final_grid_array)

def is_valid(board, row, col, num):
    # Check if 'num' can be placed at board[row][col]
    for x in range(9):
        if board[row][x] == num or board[x][col] == num or board[row - row % 3 + x // 3][col - col % 3 + x % 3] == num:
            return False
    return True

def solve_sudoku(board):
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                for num in range(1, 10):
                    if is_valid(board, i, j, num):
                        board[i][j] = num
                        if solve_sudoku(board):
                            return True
                        board[i][j] = 0  # If placing 'num' at (i, j) doesn't lead to a solution, backtrack
                return False
    return True

# Assuming 'Final_Grid' is the 9x9 Sudoku grid obtained from the image
solve_sudoku(final_grid_array)
display_sudoku_grid(np.array(final_grid_array))

