import numpy as np
import math
import time
from PIL import Image,ImageDraw
import random

def load_image_to_grayscalearray(image_path):
    image = Image.open(image_path).convert('L')
    image_array = np.array(image)
    return image_array

def image_to_array(image_path):
    image = Image.open(image_path).convert('RGB')
    image_array = np.array(image)
    return image_array

def array_to_image(array):
    array = np.array(array, dtype=np.uint8)
    return Image.fromarray(array, mode='RGB')
 
def detect_objects_grayscale(image_array, background_threshold=150):
	object_mask = image_array < background_threshold
	height, width = image_array.shape
	refined_mask = np.zeros((height, width), dtype=bool)
	for i in range(2, height - 2):
		for j in range(2, width - 2):
			#5x5 neighborhood around the pixel (i, j)
			refined_mask[i, j] = np.any(object_mask[i-2:i+3, j-2:j+3])
	return refined_mask

used_colors = set()

def linspace_custom(start, stop, num):
    if num < 2:
        raise ValueError("num must be at least 2 to create a range")
    
    step = (stop - start) / (num - 1)  # Compute the step size
    result = [start + i * step for i in range(num)]  # Generate values
    
    return result

def generate_unique_color():
    while True:
        color = (random.randint(50, 230), random.randint(50, 230), random.randint(50, 230))
        if color not in used_colors:
            used_colors.add(color)
            return list(color)

def generate_image_with_points(image, points):
    for i in range(len(points) - 1):
        x0, y0 = points[i]
        image[y0, x0] = [255, 0, 0]

    return image

def draw_circles(image, points, radius=5, color=[255,0,0]):
    img_with_circles = image.copy()
    for x, y in points:
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if 0 <= x + dx < img_with_circles.shape[0] and 0 <= y + dy < img_with_circles.shape[1]:
                    img_with_circles[x + dx, y + dy] = color 
    return img_with_circles

def neighbours(x, y, image):
    img = image
    x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
    return [
        img[x_1, y], img[x_1, y1], img[x, y1], img[x1, y1],  # P2, P3, P4, P5
        img[x1, y], img[x1, y_1], img[x, y_1], img[x_1, y_1]  # P6, P7, P8, P9
    ]

def transitions(neighbours):
    n = neighbours + neighbours[0:1] 
    return sum((n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]))

def skeletonize_object(image):
    Image_Thinned = image.copy().astype(int)
    changing1= changing2 = 1
	
    rows, cols = Image_Thinned.shape
    bridge_points = []  

    while changing1 or changing2:
        changing1 = []

        for x in range(1, rows - 1):
            for y in range(1, cols - 1):
                if Image_Thinned[x, y] == 1:
                    P2, P3, P4, P5, P6, P7, P8, P9 = n = neighbours(x, y, Image_Thinned)
                    transition_count = transitions(n)
                    neighbor_count = sum(n)
                    if (neighbor_count >= 2 and neighbor_count <= 6 and transition_count == 1 and
                        P2 * P4 * P6 == 0 and P4 * P6 * P8 == 0):
                        changing1.append((x, y))
        
        if not changing1:
            break

        for x, y in changing1:
            Image_Thinned[x, y] = 0

        changing2 = []
        for x in range(1, rows - 1):
            for y in range(1, cols - 1):
                if Image_Thinned[x, y] == 1:
                    P2, P3, P4, P5, P6, P7, P8, P9 = n = neighbours(x, y, Image_Thinned)
                    transition_count = transitions(n)
                    neighbor_count = sum(n)
                    # Thinning conditions
                    if (neighbor_count >= 2 and neighbor_count <= 6 and transition_count == 1 and
                        P2 * P4 * P8 == 0 and P2 * P6 * P8 == 0):
                        changing2.append((x, y))

        if not changing2:
            break

        for x, y in changing2:
            Image_Thinned[x, y] = 0

    return Image_Thinned

def detect_bridges(image):
    bridge_points = []
    rows, cols = image.shape

    for x in range(1, rows - 1):
        for y in range(1, cols - 1):
            if image[x, y] == 1:
                n = neighbours(x, y, image)
                neighbor_count = sum(n)
                transition_count = transitions(n)
                if transition_count >= 3 or neighbor_count == 1:  
                    bridge_points.append((x, y))

    return bridge_points

def segment_skeleton(skeleton_mask, bridge_points):
    segments = set()  
    global_visited = set() 
    
    neighbors_offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1),  (1, 0), (1, 1)
    ]
    height, width = skeleton_mask.shape
    labelled_mask = np.zeros((height, width), dtype=int)
    label=1
    def traverse_segment(start, start_bridge, restricted, labelled_mask, label):
        path = [start_bridge]
        local_visited = set()
        stack = [start]
        end_bridge = None

        while stack:
            current = stack.pop()
            if current in local_visited or current in global_visited:
                continue

            path.append(current)
            local_visited.add(current)
            global_visited.add(current)
            labelled_mask[current] = label;

            #checking if any direct neighbor is a bridge point before continuing traversal
            y, x = current
            for dy, dx in neighbors_offsets:
                ny, nx = y + dy, x + dx
                neighbor = (ny, nx)
                if neighbor in bridge_points and neighbor != start_bridge:
                    end_bridge = neighbor
                    path.append(end_bridge)
                    return end_bridge, path
            
            #exploring all neighbors only if no bridge point is encountered
            for dy, dx in neighbors_offsets:
                ny, nx = y + dy, x + dx
                neighbor = (ny, nx)
                if neighbor in restricted:
                    continue;
                if (0 <= ny < skeleton_mask.shape[0] and 0 <= nx < skeleton_mask.shape[1] 
                        and skeleton_mask[ny, nx] == 1 and neighbor not in local_visited):
                    stack.append(neighbor)
        
        return end_bridge, path
    
    for bridge in bridge_points:
        bridge_neighbors = []
        for dy, dx in neighbors_offsets:
            ny, nx = bridge[0] + dy, bridge[1] + dx
            neighbor = (ny, nx)
            if (0 <= ny < skeleton_mask.shape[0] and 0 <= nx < skeleton_mask.shape[1] 
                    and skeleton_mask[ny, nx] == 1 and neighbor not in bridge_points):
                bridge_neighbors.append(neighbor)

        for dy, dx in neighbors_offsets:
            ny, nx = bridge[0] + dy, bridge[1] + dx
            neighbor = (ny, nx)
            
            if (0 <= ny < skeleton_mask.shape[0] and 0 <= nx < skeleton_mask.shape[1] 
                    and skeleton_mask[ny, nx] == 1 and neighbor not in global_visited):
                end_bridge, path = traverse_segment(neighbor, bridge,list(set(bridge_neighbors) - {neighbor}), labelled_mask, label)
                
                if end_bridge and (bridge, end_bridge, tuple(path)) not in segments:
                    segments.add((label, bridge, end_bridge, tuple(path)))
                    label += 1;
    
    return list(segments),labelled_mask


def highlight_segments(pixels,debugimage_array, color):
	for x,y in pixels:
		debugimage_array[x,y] = color
	return debugimage_array
	
def find_knot_points(centerline_coords):
    angle_threshold = 25
    min_distance = 10
    def calculate_angle(p1, p2, p3):
    	
        #angle (in degrees) between two vectors formed by three points.
        vec1 = (p2[0] - p1[0], p2[1] - p1[1])
        vec2 = (p3[0] - p2[0], p3[1] - p2[1])

        #dot product and magnitudes
        dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
        mag1 = math.sqrt(vec1[0]**2 + vec1[1]**2)
        mag2 = math.sqrt(vec2[0]**2 + vec2[1]**2)

        if mag1 == 0 or mag2 == 0:
            return 0

        #angle in radians and convert to degrees
        cos_theta = dot_product / (mag1 * mag2)
        cos_theta = max(-1, min(1, cos_theta)) 
        angle = math.degrees(math.acos(cos_theta))

        return angle
        
    def is_far_enough(point, knot_points, min_distance):
        for kp in knot_points:
            if ((point[0] - kp[0]) ** 2 + (point[1] - kp[1]) ** 2) < min_distance ** 2:
                return False
        return True

    knot_points = [centerline_coords[0]] 

    for i in range(1, len(centerline_coords) - 1):
        p1 = centerline_coords[i - 1]
        p2 = centerline_coords[i]
        p3 = centerline_coords[i + 1]

        #angle between the two segments
        angle = calculate_angle(p1, p2, p3)

        if angle > angle_threshold and is_far_enough(p2, knot_points, min_distance):
            knot_points.append(p2)

    knot_points.append(centerline_coords[-1])  # Add the last point

    return knot_points

def explore_width(binary_mask, start_point, direction, debug_img, color):
    x, y = start_point
    distance = 0

    while True:
        x += direction[0]
        y += direction[1]
        #ix, iy = int(round(x)), int(round(y))
        if not np.isnan(x) and not np.isnan(y):
        	ix, iy = int(round(x)), int(round(y))
        else:
        	print("Warning: NaN values detected in width calculation")
        	return 0  

        if ix < 0 or ix >= binary_mask.shape[0] or iy < 0 or iy >= binary_mask.shape[1]:
            break 
        if binary_mask[ix, iy] == 0:
            break 
        
        debug_img.putpixel((iy, ix), color)
        distance += 1

    return distance

def find_width_using_skeleton(knot_points, bridge_points, binary_mask, skeleton_mask_combined, debug_img):
    def perpendicular_directions(dx, dy):
        return np.array([-dy, dx]), np.array([dy, -dx])

    widths = []

    for i in range(len(knot_points)):
        if knot_points[i] in bridge_points and sum(neighbours(knot_points[i][0], knot_points[i][1], skeleton_mask_combined)) >= 3:
        	widths.append(-1);
        	continue;

        if i == 0:  
            current_point = knot_points[i]
            next_point = knot_points[i + 1]

            dx, dy = np.array(next_point) - np.array(current_point)

        elif i == len(knot_points) - 1: 
            prev_point = knot_points[i - 1]
            current_point = knot_points[i]

            dx, dy = np.array(current_point) - np.array(prev_point)

        else: 
            prev_point = knot_points[i - 1]
            current_point = knot_points[i]
            next_point = knot_points[i + 1]

            dx, dy = np.array(next_point) - np.array(prev_point)

        perp1, perp2 = perpendicular_directions(dx, dy)
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = perp2 / np.linalg.norm(perp2)

        width1 = explore_width(binary_mask, current_point, perp1, debug_img, (255, 0, 0))  # Red
        width2 = explore_width(binary_mask, current_point, perp2, debug_img, (0, 0, 255))  # Blue

        widths.append(width1 + width2)

    return widths

def calculate_parametric_cubic_spline(points):
    n = len(points) - 1  
    if n < 1:
        raise ValueError("At least two points are required.")

    # Parameterize points by calculating distances (arc length)
    t = [0]
    for i in range(1, len(points)):
        distance = np.sqrt((points[i][0] - points[i - 1][0])**2 + (points[i][1] - points[i - 1][1])**2)
        t.append(t[-1] + distance)

    def cubic_spline_coefficients(values, t):
       
        n = len(t) - 1
        h = [t[i + 1] - t[i] for i in range(n)]
        alpha = [0] * (n + 1)
        for i in range(1, n):
            alpha[i] = (3 / h[i] * (values[i + 1] - values[i]) - 
                        3 / h[i - 1] * (values[i] - values[i - 1]))

        # Solve tridiagonal system
        l = [1] + [0] * n
        mu = [0] * (n + 1)
        z = [0] * (n + 1)

        for i in range(1, n):
            l[i] = 2 * (t[i + 1] - t[i - 1]) - h[i - 1] * mu[i - 1]  		#holds updated diagonal elements (To make diagonal entry equals to 1, we need to do l[i]/l[i])
            mu[i] = h[i] / l[i] 	#Holds updated diagonal's RHS side entries and factor by which lower diagonal entry evaluates
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]  				#RHS

        l[n] = 1
        z[n] = 0

        # Back substitution
        c = [0] * (n + 1)
        for j in range(n - 1, -1, -1):
            c[j] = z[j] - mu[j] * c[j + 1]

        b = [0] * n
        d = [0] * n
        a = [values[i] for i in range(n)]

        for i in range(n):
            b[i] = ((values[i + 1] - values[i]) / h[i] -
                    h[i] * (c[i + 1] + 2 * c[i]) / 3)
            d[i] = (c[i + 1] - c[i]) / (3 * h[i])

        coefficients = []
        for i in range(n):
            coefficients.append({
                'a': a[i],
                'b': b[i],
                'c': c[i],
                'd': d[i],
                't_i': t[i]
            })

        return coefficients

    x_values = [p[0] for p in points]
    y_values = [p[1] for p in points]

    x_coefficients = cubic_spline_coefficients(x_values, t)
    y_coefficients = cubic_spline_coefficients(y_values, t)
    return {'x': x_coefficients, 'y': y_coefficients, 't': t}


def evaluate_spline_width(spline_coefficients, t_values, knot_widths):
    object_points = [] 
    x_segments = spline_coefficients['x']
    y_segments = spline_coefficients['y']
    
    for t in t_values:
        for i in range(len(x_segments)): 
            t_i = x_segments[i]['t_i']
            if t_i <= t <= (x_segments[i+1]['t_i'] if i + 1 < len(x_segments) else t_i):
                dx = t - t_i
                
                x = (x_segments[i]['a'] +
                     x_segments[i]['b'] * dx +
                     x_segments[i]['c'] * dx**2 +
                     x_segments[i]['d'] * dx**3)
                
                y = (y_segments[i]['a'] +
                     y_segments[i]['b'] * dx +
                     y_segments[i]['c'] * dx**2 +
                     y_segments[i]['d'] * dx**3)

                # Compute first derivatives dx/dt and dy/dt (tangent)
                dx_dt = (x_segments[i]['b'] +
                         2 * x_segments[i]['c'] * dx +
                         3 * x_segments[i]['d'] * dx**2)
                
                dy_dt = (y_segments[i]['b'] +
                         2 * y_segments[i]['c'] * dx +
                         3 * y_segments[i]['d'] * dx**2)

                normal = np.array([-dy_dt, dx_dt])
                normal_length = np.linalg.norm(normal)
                
                if normal_length != 0:
                    normal /= normal_length  

                segment_width = (knot_widths[i] + knot_widths[i+1]) / 2

                half_width = segment_width / 2
                left_x = x + half_width * normal[0]
                left_y = y + half_width * normal[1]
                right_x = x - half_width * normal[0]
                right_y = y - half_width * normal[1]

                object_points.append((left_x, left_y))
                object_points.append((right_x, right_y))
                
                num_steps = max(1, int(half_width)) 
                for step in range(1, num_steps):
                    factor = step / num_steps
                    mid_x1 = x + factor * half_width * normal[0]
                    mid_y1 = y + factor * half_width * normal[1]
                    mid_x2 = x - factor * half_width * normal[0]
                    mid_y2 = y - factor * half_width * normal[1]
                    object_points.append((mid_x1, mid_y1))
                    object_points.append((mid_x2, mid_y2))
                
                break 

    return object_points

def evaluate_spline(spline_coefficients, t_values):
    centerline_points = [] 
    x_segments = spline_coefficients['x']
    y_segments = spline_coefficients['y']
    
    for t in t_values:
        for i in range(len(x_segments)): 
            t_i = x_segments[i]['t_i']
            if t_i <= t <= (x_segments[i+1]['t_i'] if i + 1 < len(x_segments) else t_i):
                
                dx = t - t_i
                
                x = (x_segments[i]['a'] +
                     x_segments[i]['b'] * dx +
                     x_segments[i]['c'] * dx**2 +
                     x_segments[i]['d'] * dx**3)
                
                y = (y_segments[i]['a'] +
                     y_segments[i]['b'] * dx +
                     y_segments[i]['c'] * dx**2 +
                     y_segments[i]['d'] * dx**3)
                
                centerline_points.append((x, y))
                break 

    return centerline_points
def percentile(data, percent):
    data = sorted(data)
    k = (len(data) - 1) * (percent / 100)
    f = int(k)
    c = f + 1 if f + 1 < len(data) else f
    return data[f] + (k - f) * (data[c] - data[f])

def adjust_width(arr):
    """Replaces -1 at the start or end with the nearest valid value found within range."""
    valid_vals = [x for x in arr if x != -1]
    if not valid_vals:
        return arr 

    q1 = percentile(valid_vals, 25)
    q3 = percentile(valid_vals, 75)
    iqr = q3 - q1
    lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
    
    def in_valid_range(x):
        return lower <= x <= upper

    if arr[0] == -1:
        for i in range(1, len(arr)):
            if in_valid_range(arr[i]):
                arr[:i] = [arr[i]] * i 
                break

    if arr[-1] == -1:
        for i in range(len(arr) - 2, -1, -1):
            if in_valid_range(arr[i]):
                arr[i + 1:] = [arr[i]] * (len(arr) - i - 1)
                break

    return arr

def main():
    image_path = 'input/sample2.jpg'
    image_array = load_image_to_grayscalearray(image_path)
    object_mask = detect_objects_grayscale(image_array, background_threshold=150)
   
    skeleton_mask_combined = skeletonize_object(object_mask)
    bridge_points = detect_bridges(skeleton_mask_combined)
    print(f"Total number of bridge points: {len(bridge_points)}\nbridge_points: {bridge_points}")
    skeleton_rgb = np.stack([skeleton_mask_combined * 255] * 3, axis=-1)

    Image_With_Markers = draw_circles(skeleton_rgb, bridge_points, radius=5, color=[255,0,0])
    
    skeleton_image = array_to_image(skeleton_rgb)
    skeleton_image.save('output/skeletonized_all_objects.png')
    detected_circles = array_to_image(Image_With_Markers)
    detected_circles.save('output/detected_bridges.png')

    segments, labelled_mask = segment_skeleton(skeleton_mask_combined, bridge_points)
    unique_labels, counts = np.unique(labelled_mask, return_counts=True)
    print("Pixel count for each label:\n")
    print("\t".join(f"{count} pixels" for label, count in zip(unique_labels, counts)))
    print("No of segments: ", len(segments))
    print("Unique labels assigned:", unique_labels)

    height, width = image_array.shape
    debugimage_array = np.ones((height, width, 3), dtype=np.uint8) * 255
    all_knot_points = []
    all_spline_segments = []  
    width_list = []  
    height, width = object_mask.shape
    final_reconstructed_image = np.zeros((height*2, width*2, 3), dtype=np.uint8)  
    debug_img = Image.open(image_path).convert('RGB')
    for object_no, start_bridge, end_bridge, ordered_pixels in segments:
    	print(f"Segment from {start_bridge} to {end_bridge} with {len(ordered_pixels)} pixels.")
    	random_color = generate_unique_color()
    	debugimage_array = highlight_segments(ordered_pixels, debugimage_array, color= random_color)
    	knot_points = find_knot_points(ordered_pixels)
    	all_knot_points.extend(knot_points)

    	width_value = find_width_using_skeleton(knot_points, bridge_points,  object_mask, skeleton_mask_combined, debug_img)
    	print(width_value)
    	width_value = adjust_width(width_value)
        		
    	print(width_value)

    	swapped_array = [(y, x) for (x, y) in knot_points]
    	spline_coefficients = calculate_parametric_cubic_spline(swapped_array)
    	all_spline_segments.append(spline_coefficients)

    	t_min, t_max = spline_coefficients['t'][0], spline_coefficients['t'][-1]
    	t_values = linspace_custom(t_min, t_max, 100)
    	evaluated_points = evaluate_spline(spline_coefficients, t_values)
    	evaluated_points = evaluate_spline_width(spline_coefficients, t_values, width_value)
    	clean_points = [(int(round(x)), int(round(y))) for x, y in evaluated_points]

    	reconstructed_image_array = generate_image_with_points(final_reconstructed_image, clean_points)

    debug_image_segments = array_to_image(debugimage_array)
    debug_image_segments.save('output/debug_Segments.png')
    reconstructed_image = array_to_image(final_reconstructed_image)
    reconstructed_image.save('output/reconstructed_image.png')

    skeleton_rgb = np.stack([skeleton_mask_combined * 255] * 3, axis=-1)
    for (x, y) in all_knot_points:
        skeleton_rgb[x, y] = [255, 0, 0]
    
    skeleton_image = array_to_image(skeleton_rgb)
    skeleton_image.save('output/knot_points.png')

    debug_img.save('output/debug_width_exploration.png')

if __name__ == "__main__":
    main()
