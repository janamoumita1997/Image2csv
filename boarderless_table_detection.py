import os
import cv2
import pytesseract
import pandas as pd

# This only works if there's only one table on a page
# Important parameters:
#  - morph_size
#  - min_text_height_limit
#  - max_text_height_limit
#  - cell_threshold
#  - min_columns


def pre_process_image(img, save_in_file, morph_size=(12, 5)):

    # get rid of the color
    pre = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Otsu threshold
    pre = cv2.threshold(pre, 250, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # dilate the text to make it solid spot
    cpy = pre.copy()
    struct = cv2.getStructuringElement(cv2.MORPH_RECT, morph_size)
    cpy = cv2.dilate(~cpy, struct, anchor=(-1, -1), iterations=1)
    pre = ~cpy
    # print("pre >> ",pre)
    if save_in_file is not None:
        cv2.imwrite(save_in_file, pre)
    return pre


def find_text_boxes(pre, min_text_height_limit=10, max_text_height_limit=40):
    # Looking for the text spots contours
    # OpenCV 3
    # img, contours, hierarchy = cv2.findContours(pre, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # OpenCV 4
    contours, hierarchy = cv2.findContours(pre, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Getting the texts bounding boxes based on the text size assumptions
    boxes = []
    for contour in contours:
        # print("contour >> ",contour)
        box = cv2.boundingRect(contour)
        # print("box >>> ",box)
        h = box[3]

        if min_text_height_limit < h < max_text_height_limit:
            boxes.append(box)
    # print("boxes >> ",boxes)
    return boxes


def find_table_in_boxes(boxes, cell_threshold=10, min_columns=1):
    rows = {}
    cols = {}

    # Clustering the bounding boxes by their positions
    for box in boxes:
        (x, y, w, h) = box
        col_key = x // cell_threshold
        row_key = y // cell_threshold
        cols[row_key] = [box] if col_key not in cols else cols[col_key] + [box]
        rows[row_key] = [box] if row_key not in rows else rows[row_key] + [box]

    # print("rows : ",rows)
    # print("cols : ",cols)
    # print("\n")
    # Filtering out the clusters having less than 2 cols
    table_cells = list(filter(lambda r: len(r) >= min_columns, rows.values()))
    # Sorting the row cells by x coord
    table_cells = [list(sorted(tb)) for tb in table_cells]
    # print("\n\n")
    # print("table cells : ",table_cells)
    # Sorting rows by the y coord
    table_cells = list(sorted(table_cells, key=lambda r: r[0][1]))
    # print("\n\n")
    # print("table cells after sorting : ",table_cells)

    table_cells = list(sorted(table_cells, key=lambda r: r[0][0]>= 300))
    # print("\n\n")
    # print("table cells after sorting : ",table_cells)
    # table_cells = table_cells[:8]


    return table_cells


def build_lines(table_cells):
    if table_cells is None or len(table_cells) <= 0:
        return [], []

    max_last_col_width_row = max(table_cells, key=lambda b: b[-1][2])
    print("max_last_col_width_row >>>>> ",max_last_col_width_row)
    max_x = max_last_col_width_row[-1][0] + max_last_col_width_row[-1][2]

    max_last_row_height_box = max(table_cells[-1], key=lambda b: b[3])
    print("max_last_row_height_box >>>> ",max_last_row_height_box)
    max_y = max_last_row_height_box[1] + max_last_row_height_box[3]

    hor_lines = []
    ver_lines = []

    for box in table_cells:
        x = box[0][0]
        y = box[0][1]
        if x <= 300:
            hor_lines.append((x-5, y, max_x+5, y))

    for box in table_cells[0]:
        x = box[0]
        y = box[1]
        if x <= 300:
            ver_lines.append((x-3, y, x-8, max_y))

    (x, y, w, h) = table_cells[0][-1]
    ver_lines.append((max_x, y-5, max_x, max_y))
    (x, y, w, h) = table_cells[0][0]
    hor_lines.append((x-4, max_y, max_x, max_y))

    print("hor_lines : ",hor_lines)
    print("\n")
    print("ver_lines : ",ver_lines)
    return hor_lines, ver_lines

def boundary_box(hor_line):
    x1 = hor_line[1]
    y = []
    for each_line in hor_line[1:]:
        z = []
        x2 = each_line
        hight_check = int(x2[1]) - int(x1[1])
        if hight_check >= 50:
            z.append(x1)
            z.append(x2)
        if len(z)!=0:
            y.append(z)

        x1 = x2

    return y

if __name__ == "__main__":
    in_file = os.path.join("data", "/home/oem/manna/Assignments/assignment_1/data_sample_page-0001.jpg")
    pre_file = os.path.join("data", "/home/oem/manna/Assignments/assignment_1/output/pre_data_sample_page-0001.png")
    out_file = os.path.join("data", "/home/oem/manna/Assignments/assignment_1/output/out_data_sample_page-0001.png")

    img = cv2.imread(os.path.join(in_file))

    pre_processed = pre_process_image(img, pre_file)
    text_boxes = find_text_boxes(pre_processed)
    cells = find_table_in_boxes(text_boxes)
    hor_lines, ver_lines = build_lines(cells)
    boundary_box_coordinate = boundary_box(hor_lines)

    # Visualize the result
    vis = img.copy()

    # for box in text_boxes:
    #     (x, y, w, h) = box
    #     cv2.rectangle(vis, (x, y), (x + w - 2, y + h - 2), (0, 255, 0), 2)

    # for line in hor_lines:
    #     [x1, y1, x2, y2] = line
    #     cv2.line(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # for line in ver_lines:
    #     [x1, y1, x2, y2] = line
    #     cv2.line(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
    Field_name = []
    Value = []
    des = []
    for i in boundary_box_coordinate:
        x1 = i[0][0]
        y1 = i[0][1]
        x2 = i[1][2]
        y2 = i[1][3]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
        crop_img = vis[y1:y2, x1:x2]
        text = pytesseract.image_to_string(crop_img, lang='eng', config='--psm 6')

        Description = "\n".join(text.split("\n") [1:])
        f = text.split("\n")[0]
        field_name = f.split()[0]
        value = f.split()[1].replace("$",'s').replace("}",')')
        print("field name : ",field_name)
        print("value : ",value)
        print("Description : ",Description)

        Field_name.append(field_name)
        Value.append(value)
        des.append(Description)
        # cv2.imshow("cropped", crop_img)
        # cv2.waitKey(0)
    data = {'Field_name':Field_name,'Value':Value,'Description':des}
    df = pd.DataFrame(data)
    print("df >>> ",df)
    # df.reset_index(drop=True)
    df.to_csv("output.csv", index=False)
    cv2.imwrite(out_file, vis)