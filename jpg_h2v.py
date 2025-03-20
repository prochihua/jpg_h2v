import cv2
import numpy as np

class Element:
    def __init__(self):
        self.x = -1
        self.y = -1
        self.width = -1
        self.height = 0
        self.column_idx = 0
        self.char_idx = 0

class Column:
    def __init__(self):
        self.start = -1
        self.end = -1
        self.elements = []

def split_column(mat):
    """垂直投影分割，找出直排文本的各列"""
    columns = []
    if mat is None or mat.size == 0:
        return columns
    
    # 統計每一列的黑色像素數量
    pos = [0] * mat.shape[1]
    for col in range(mat.shape[1]):
        for row in range(mat.shape[0]):
            if mat[row, col] == 0:
                pos[col] += 1
    
    # 繪製投影結果
    result = np.ones((mat.shape[0], mat.shape[1]), dtype=np.uint8) * 255
    for col in range(result.shape[1]):
        size = pos[col]
        if size > 0:
            cv2.line(result, (col, 0), (col, size), (0, 0, 0), 1)
    
    cv2.imwrite("1_column_projection.jpg", result)
    
    # 根據投影結果分割出各列
    column = Column()
    for i in range(mat.shape[1]):
        if pos[i] > 0:
            if column.start == -1:
                column.start = i
        elif pos[i] == 0:
            if column.start > -1 and column.end == -1:
                column.end = i
                columns.append(column)
                column = Column()
    
    # 處理最後一列
    if column.start > -1 and column.end == -1:
        column.end = mat.shape[1] - 1
        columns.append(column)
    
    return columns

def split_character(column, mat, draw_mat):
    """水平投影分割，找出每列中的各個字符"""
    elements = []
    pos = [0] * mat.shape[0]
    
    # 統計指定列範圍內，每一行的黑色像素數量
    for r in range(mat.shape[0]):
        for c in range(column.start, min(column.end, mat.shape[1])):
            if mat[r, c] == 0:
                pos[r] += 1
    
    # 繪製投影結果（在調試圖像上）
    for r in range(draw_mat.shape[0]):
        size = pos[r]
        if size > 0:
            try:
                cv2.line(draw_mat, (column.end, r), (column.end + size, r), (0, 0, 255), 1)
            except:
                pass
    
    # 根據投影結果分割出各個字符
    element = Element()
    for i in range(mat.shape[0]):
        if pos[i] > 0:
            if element.x == -1 and element.y == -1:
                element.x = column.start
                element.y = i
                element.width = column.end - column.start
        elif pos[i] == 0:
            if element.x > -1 and element.height == 0:
                element.height = i - element.y
                #if element.height > 5:  # 忽略太小的區域，可能是噪點
                elements.append(element)
                element = Element()
    
    # 處理最後一個字符
    if element.x > -1 and element.height == 0:
        element.height = mat.shape[0] - 1 - element.y
        #if element.height > 5:  # 忽略太小的區域
        elements.append(element)
    
    # 在每列的最後一個字符後加入一個特殊標記，表示需要換行
    if elements:
        newline_marker = Element()
        newline_marker.x = -1  # 使用特殊值來標記換行符號
        newline_marker.y = -1
        newline_marker.width = -1
        newline_marker.height = -1
        elements.append(newline_marker)
        
    return elements

def merge_small_characters(characters, avg_height, avg_width):
    """
    合併異常小的字符，處理被錯誤分割的情況
    比較上下相鄰區域，將小字塊合併到較小的一側
    """
    if len(characters) <= 1:
        return characters
    
    # 設置大小閾值，小於平均值的一定比例認為是被錯誤分割的字符
    height_threshold = avg_height * 0.6  # 降低閾值以捕獲更多的小字塊
    width_threshold = avg_width * 0.7
    
    # 按照y坐標排序
    characters.sort(key=lambda x: x.y)
    
    # 創建字符索引的副本，用於跟蹤合併操作
    char_indices = list(range(len(characters)))
    merged_status = [False] * len(characters)
    
    for i in range(len(characters)):
        # 如果已被合併，跳過
        if merged_status[i]:
            continue
            
        current_char = characters[i]
        
        # 如果不是小字塊，跳過
        if current_char.height >= height_threshold:
            continue
        
        # 尋找上方和下方的相鄰字塊
        upper_idx = -1
        lower_idx = -1
        upper_distance = float('inf')
        lower_distance = float('inf')
        
        for j in range(len(characters)):
            if i == j or merged_status[j]:
                continue
                
            other_char = characters[j]
            
            # 檢查x軸方向是否重疊（在同一列）
            x_overlap = (min(current_char.x + current_char.width, other_char.x + other_char.width) - 
                         max(current_char.x, other_char.x))
            if x_overlap <= 0:
                continue
                
            # 上方字塊
            if other_char.y + other_char.height <= current_char.y:
                dist = current_char.y - (other_char.y + other_char.height)
                if dist < upper_distance and dist < avg_height * 0.5:
                    upper_distance = dist
                    upper_idx = j
            
            # 下方字塊
            elif other_char.y >= current_char.y + current_char.height:
                dist = other_char.y - (current_char.y + current_char.height)
                if dist < lower_distance and dist < avg_height * 0.5:
                    lower_distance = dist
                    lower_idx = j
        
        # 決定合併方向
        merge_idx = -1
        if upper_idx >= 0 and lower_idx >= 0:
            # 比較上下字塊的大小，合併到較小的一方
            upper_size = characters[upper_idx].width * characters[upper_idx].height
            lower_size = characters[lower_idx].width * characters[lower_idx].height
            
            merge_idx = upper_idx if upper_size <= lower_size else lower_idx
        elif upper_idx >= 0:
            merge_idx = upper_idx
        elif lower_idx >= 0:
            merge_idx = lower_idx
        
        # 執行合併
        if merge_idx >= 0:
            target_char = characters[merge_idx]
            
            # 創建合併後的字符
            merged = Element()
            merged.x = min(current_char.x, target_char.x)
            merged.y = min(current_char.y, target_char.y)
            merged.width = max(current_char.x + current_char.width, 
                              target_char.x + target_char.width) - merged.x
            merged.height = max(current_char.y + current_char.height, 
                               target_char.y + target_char.height) - merged.y
            merged.column_idx = current_char.column_idx
            merged.char_idx = min(current_char.char_idx, target_char.char_idx)
            
            # 更新字符列表
            characters[merge_idx] = merged
            merged_status[i] = True
    
    # 返回未被合併的字符和合併後的字符
    result = [char for i, char in enumerate(characters) if not merged_status[i]]
    
    return result

def rearrange_vertical_to_horizontal(binary_image, src_image):
    """將直排文本重排為橫排文本，考慮合理的排版，並處理字符分割問題"""
    # 分割列
    columns = split_column(binary_image)
    print(f"找到 {len(columns)} 列")
    
    # 創建調試圖像
    debug_img = src_image.copy()
    merged_debug = src_image.copy()  # 用於顯示合併後的結果
    
    # 存儲所有檢測到的字符
    all_chars = []
    column_chars = []
    
    # 先收集所有字符信息，計算平均尺寸
    for i, column in enumerate(columns):
        characters = split_character(column, binary_image, debug_img)
        print(f"第 {i+1} 列找到 {len(characters)} 個字符")
        
        # 在調試圖像上標記原始分割的字符
        for j, char in enumerate(characters):
            cv2.rectangle(debug_img, (char.x, char.y), 
                         (char.x + char.width, char.y + char.height), 
                         (0, 0, 255), 1)
            char.column_idx = i
            char.char_idx = j
        
        column_chars.append(characters)
        all_chars.extend(characters)
    
    if not all_chars:
        print("未檢測到任何字符，請檢查圖像質量和閾值設置")
        return src_image, debug_img
    
    # 計算字符平均尺寸
    avg_height = sum(char.height for char in all_chars) / len(all_chars)
    avg_width = sum(char.width for char in all_chars) / len(all_chars)
    
    print(f"字符平均尺寸: 寬度={avg_width:.2f}, 高度={avg_height:.2f}")
    
    # 對每列的字符進行合併處理
    merged_all_chars = []
    for i, chars in enumerate(column_chars):
        merged_chars = merge_small_characters(chars, avg_height, avg_width)
        print(f"第 {i+1} 列: 原有 {len(chars)} 個字符，合併後 {len(merged_chars)} 個字符")
        
        # 標記合併後的字符
        for j, char in enumerate(merged_chars):
            cv2.rectangle(merged_debug, (char.x, char.y), 
                         (char.x + char.width, char.y + char.height), 
                         (0, 255, 0), 1)
        
        merged_all_chars.extend(merged_chars)
    
    # 保存調試圖像
    cv2.imwrite("detected_chars_original.jpg", debug_img)
    cv2.imwrite("detected_chars_merged.jpg", merged_debug)
    
    # 按照從右到左的列順序，從上到下的字符順序排序
    # 中文直排通常從右向左閱讀
    merged_all_chars.sort(key=lambda x: (-x.column_idx, x.char_idx))
    
    # 重新計算合併後的平均尺寸
    if merged_all_chars:
        avg_height = sum(char.height for char in merged_all_chars) / len(merged_all_chars)
        avg_width = sum(char.width for char in merged_all_chars) / len(merged_all_chars)
    
    # 計算排版參數
    char_padding = int(avg_width * 0.2)
    line_padding = int(avg_height * 0.5)
    
    # 計算合適的每行字符數
    chars_per_line = max(5, int(len(merged_all_chars) / len(columns)))
    
    # 修改部分開始 - 使用預設大底圖然後裁剪的方法
    # 計算新圖像的寬度 - 仍然需要確定合適的寬度
    new_width = int(chars_per_line * (avg_width + char_padding) + char_padding)
    
    # 創建一個足夠大的白色背景圖像，高度預設為原圖的3倍（通常足夠容納所有文字）
    max_height = int(src_image.shape[0] * 3)
    horizontal_img = np.ones((max_height, new_width, 3), dtype=np.uint8) * 255
    
    print(f"創建臨時底圖，尺寸: {new_width}x{max_height}，每行字符數: {chars_per_line}")
    # 修改部分結束
    
    # 放置字符
    current_line = 0
    current_pos = 0
    
    for char in merged_all_chars:
        try:
            if char.x == -1 and char.y == -1:
                # 遇到換行符號，換行
                current_pos = 0
                current_line += 1
                continue
            # 從原圖提取字符
            y_start = max(0, char.y)
            y_end = min(src_image.shape[0], char.y + char.height)
            x_start = max(0, char.x)
            x_end = min(src_image.shape[1], char.x + char.width)
            
            if y_end <= y_start or x_end <= x_start:
                continue  # 跳過無效區域
                
            char_img = src_image[y_start:y_end, x_start:x_end].copy()
            
            # 計算在新圖像中的位置
            new_x = int(current_pos * (avg_width + char_padding) + char_padding)
            new_y = int(current_line * (avg_height + line_padding) + line_padding)
            
            # 確保不超出邊界
            if new_x + char.width > new_width:
                current_pos = 0
                current_line += 1
                new_x = int(current_pos * (avg_width + char_padding) + char_padding)
                new_y = int(current_line * (avg_height + line_padding) + line_padding)
            
            if new_y + char.height > max_height:  # 修改為檢查max_height
                print(f"警告：超出預設最大高度，字符無法完全顯示")
                break
                
            # 確保目標區域在圖像範圍內
            target_height = min(char_img.shape[0], max_height - new_y)  # 修改為使用max_height
            target_width = min(char_img.shape[1], new_width - new_x)
            
            if target_height <= 0 or target_width <= 0:
                continue
                
            # 調整小字符的居中位置
            adjusted_y = new_y
            adjusted_x = new_x
            if char.height < avg_height * 0.7:  # 如果字符高度小於平均高度的70%，垂直居中
                adjusted_y += int((avg_height - char.height) / 2)
            if char.width < avg_width * 0.7:  # 如果字符寬度小於平均寬度的70%，水平居中
                adjusted_x += int((avg_width - char.width) / 2)
        
            # 放置字符
            horizontal_img[adjusted_y:adjusted_y+target_height, adjusted_x:adjusted_x+target_width] = char_img[:target_height, :target_width]
            
            # 更新位置
            current_pos += 1
            if current_pos >= chars_per_line:
                current_pos = 0
                current_line += 1
        
        except Exception as e:
            print(f"處理字符時發生錯誤: {e}")
            continue
    
    # 添加部分 - 裁剪圖像到實際使用的高度
    # 計算實際使用的高度（最後一個字符所在行的底部加上一些額外空間）
    actual_height = int(current_line * (avg_height + line_padding) + line_padding + avg_height)
    
    # 確保實際高度不超過預設的最大高度
    actual_height = min(actual_height, max_height)
    
    # 裁剪圖像
    horizontal_img = horizontal_img[:actual_height, :, :]
    
    print(f"裁剪後的圖像尺寸: {horizontal_img.shape[1]}x{horizontal_img.shape[0]}")
    # 添加部分結束
    
    # 保存橫排圖像
    cv2.imwrite("horizontal_text.jpg", horizontal_img)
    
    return horizontal_img, merged_debug

def main():
    # 讀取原始圖像
    input_path = "./images/0001.jpg"  # 替換為您的輸入圖像路徑
    src = cv2.imread(input_path)
    
    if src is None:
        print(f"無法讀取圖像: {input_path}")
        return
    
    print(f"原始圖像尺寸: {src.shape}")
    
    # 轉為灰度圖
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    
    # 二值化處理
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # 保存二值化圖像用於調試
    cv2.imwrite("binary.jpg", binary)
    
    # 直排轉橫排處理
    horizontal_img, debug_img = rearrange_vertical_to_horizontal(binary, src)
    
    # 顯示結果
    cv2.imwrite("debug_visualization.jpg", debug_img)
    cv2.imwrite("horizontal_result.jpg", horizontal_img)
    
    print("處理完成，結果已保存")

if __name__ == "__main__":
    main()
