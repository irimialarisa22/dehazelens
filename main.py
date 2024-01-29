from flask import Flask, request, jsonify
import cv2
import numpy as np
import math

app = Flask(__name__)


def compute_dark_channel(image, window_size):
    blue, green, red = cv2.split(image)
    dark_channel = cv2.min(cv2.min(red, green), blue)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    dark_channel_min = cv2.erode(dark_channel, kernel)
    return dark_channel_min


def estimate_atmospheric_light(image, dark_channel_min):
    [height, width] = image.shape[:2]
    image_size = height * width
    num_pixels = int(max(math.floor(image_size / 1000), 1))
    dark_vector = dark_channel_min.reshape(image_size)
    image_vector = image.reshape(image_size, 3)

    sorted_indices = dark_vector.argsort()
    sorted_indices = sorted_indices[image_size - num_pixels::]

    atmospheric_sum = np.zeros([1, 3])
    for index in range(1, num_pixels):
        atmospheric_sum = atmospheric_sum + image_vector[sorted_indices[index]]

    atmospheric_light = atmospheric_sum / num_pixels
    return atmospheric_light


def estimate_transmission(image, atmospheric_light, window_size):
    omega = 0.95
    normalized_image = np.empty(image.shape, image.dtype)

    for channel in range(0, 3):
        normalized_image[:, :, channel] = image[:, :, channel] / atmospheric_light[0, channel]

    transmission = 1 - omega * compute_dark_channel(normalized_image, window_size)
    return transmission


def guided_filter(image, p, window_size, epsilon):
    mean_image = cv2.boxFilter(image, cv2.CV_64F, (window_size, window_size))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (window_size, window_size))
    mean_image_p = cv2.boxFilter(image * p, cv2.CV_64F, (window_size, window_size))
    covariance_image_p = mean_image_p - mean_image * mean_p

    mean_image_squared = cv2.boxFilter(image * image, cv2.CV_64F, (window_size, window_size))
    variance_image = mean_image_squared - mean_image * mean_image

    a = covariance_image_p / (variance_image + epsilon)
    b = mean_p - a * mean_image

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (window_size, window_size))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (window_size, window_size))

    result = mean_a * image + mean_b
    return result


def refine_transmission(image, estimated_transmission):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = np.float64(gray_image) / 255
    radius = 60
    epsilon = 0.0001
    refined_transmission = guided_filter(gray_image, estimated_transmission, radius, epsilon)

    return refined_transmission


def recover_scene_radiance(image, estimated_transmission, atmospheric_light, min_transmission=0.1):
    result = np.empty(image.shape, image.dtype)
    estimated_transmission = cv2.max(estimated_transmission, min_transmission)

    for channel in range(0, 3):
        result[:, :, channel] = (image[:, :, channel] - atmospheric_light[0, channel]) / estimated_transmission + \
                                atmospheric_light[0, channel]

    return result


def dehaze_image(input_image_path):
    normalized_input_image = input_image_path.astype('float64') / 255
    estimated_dark_channel = compute_dark_channel(normalized_input_image, 15)
    estimated_atmospheric_light = estimate_atmospheric_light(normalized_input_image, estimated_dark_channel)
    estimated_transmission = estimate_transmission(normalized_input_image, estimated_atmospheric_light, 15)
    refined_transmission = refine_transmission(input_image_path, estimated_transmission)
    dehazed_image = recover_scene_radiance(normalized_input_image, refined_transmission, estimated_atmospheric_light,
                                           0.1)

    return dehazed_image


@app.route('/api/dehaze-image', methods=['POST'])
def dehaze_image_endpoint():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        uploaded_file = request.files['image']
        image_data = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
        input_image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

        dehazed_image_bytes = dehaze_image(input_image)

        _, encoded_image = cv2.imencode('.png', (dehazed_image_bytes * 255).astype(np.uint8))
        dehazed_image_bytes = encoded_image.tobytes()

        return dehazed_image_bytes, 200, {'Content-Type': 'image/png'}
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
