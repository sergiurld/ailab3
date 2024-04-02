import math
import random

from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
from array import array
import os
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import sys
import time
import textdistance

'''
Authenticate
Authenticates your credentials and creates a client.
'''
subscription_key = os.environ["VISION_KEY"]
endpoint = os.environ["VISION_ENDPOINT"]
computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))
'''
END - Authenticate
'''


def process_image(image_path, computer_vision_client):
    try:
        # Open the image file in binary read mode
        with open(image_path, "rb") as image_file:
            # Read the image in stream mode with the Computer Vision client
            read_response = computer_vision_client.read_in_stream(
                image=image_file,
                mode="Printed",
                raw=True
            )
            # Extract the operation ID from the response headers
            operation_id = read_response.headers['Operation-Location'].split('/')[-1]

            # Wait for the read operation to complete
            while True:
                read_result = computer_vision_client.get_read_result(operation_id)
                if read_result.status not in ['notStarted', 'running']:
                    break
                time.sleep(1)

            # Process the result if the operation succeeded
            if read_result.status == OperationStatusCodes.succeeded:
                result = ' '.join(
                    line.text for text_result in read_result.analyze_result.read_results for line in text_result.lines)
            else:
                result = None

    except Exception as e:
        print("An error occurred:", e)
        result = None

    return result


def WORD_ERROR_RATE(result, text):
    result_words = result.split()
    text_words = text.split()
    wer = sum(i != j for i, j in zip(result_words, text_words))
    return wer / len(result_words)



def CHARACTER_ERROR_RATE(result, text):
    nr_total = len(text)
    erori = 0

    for gt_char, result_char in zip(text, result):
        if gt_char != result_char:
            erori += 1

    return erori / nr_total




def compute_hamming_distance(text: str, result: str) -> int:
    if len(text) != len(result):
        raise ValueError("Strings must have equal length to compute Hamming distance")

    # Initialize the Hamming distance
    hamming_distance = 0

    # Iterate over the characters of the strings and compare them
    for gt_char, result_char in zip(text, result):
        if gt_char != result_char:
            hamming_distance += 1

    return hamming_distance


def compute_jaro_winkler_distance(result, ground_truth):
    if len(ground_truth) != len(result):
        raise ValueError("Strings must have equal length to compute Jaro-Winkler distance")

    # Compute the Jaro-Winkler distance between the two strings
    distance = textdistance.jaro_winkler(result, ground_truth)

    return distance


def EX1(path, text, client):
    result = process_image(path, client)
    rating_wer = WORD_ERROR_RATE(result, text)
    rating_cer = CHARACTER_ERROR_RATE(result, text)
    try:
        rating_ham = compute_hamming_distance(result, text)
    except ValueError:
        rating_ham = math.inf
    try:
        rating_jaro_winkler = compute_jaro_winkler_distance(result, text)
    except ValueError:
        rating_jaro_winkler = math.inf
    print("Rezultatul este : " + result)
    print("Textul original este : " + text)
    print(f"WORD ERROR RATE: {rating_wer}")
    print(f"CHARACTER WORD RATE: {rating_cer}")
    print(f"Hamming distance character: {rating_ham}")
    print(f"Jaro-Winkler distance words : {rating_jaro_winkler}")




def read_bounding_boxes(border: str) -> list[list[tuple[float, float]]]:
    bboxes = []
    with open(border, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            line = line.replace('[', '').replace(']', '').replace('(', '').replace(')', '')
            coords = line.split(', ')
            bbox = [(float(coords[i]), float(coords[i + 1])) for i in range(0, len(coords), 2)]
            bboxes.append(bbox)
    return bboxes


def EX2(image: str, border: str, client: ComputerVisionClient) -> list[list[tuple[float, float]]]:
    img = open(image, "rb")

    bboxes = read_bounding_boxes(border)

    read_response = client.read_in_stream(
        image=img,
        mode="Printed",
        raw=True
    )
    operation_id = read_response.headers['Operation-Location'].split('/')[-1]
    while True:
        read_result = client.get_read_result(operation_id)
        if read_result.status not in ['notStarted', 'running']:
            break
        time.sleep(1)

    detected_borders = []
    if read_result.status == OperationStatusCodes.succeeded:
        for text_result in read_result.analyze_result.read_results:
            imagine = Image.open(image)
            draw = ImageDraw.Draw(imagine)
            coverage=0
            for line, actual_border in zip(text_result.lines, bboxes):
                colturi = [(line.bounding_box[i], line.bounding_box[i + 1]) for i in
                           range(0, len(line.bounding_box), 2)] #
                draw.polygon(colturi, outline="purple")#desenez poligonul
                actual_text_size = (actual_border[1][0] - actual_border[0][0]) * (
                            actual_border[2][1] - actual_border[0][1])# calculez aria reala
                ai_text_size = (colturi[1][0] - colturi[0][0]) * (colturi[2][1] - colturi[0][1])#calculez aria AI

                coverage += ai_text_size / actual_text_size
                detected_borders.append(colturi)
            coverage /= len(bboxes)  # normalize by number of lines
            print(f"IA a acoperit localizat in proportie de {coverage * 100:.2f}% textul din imagine")
            imagine.show()

    return detected_borders



def EX3(image_path,text,client):
    x = random.randint(0, 1000000)
    y = random.randint(0, 1000000)
    img = Image.open(image_path)
    new_image = img.resize((300, 300))
    new_image.save("C:\\Users\\sergi\\PycharmProjects\\AILAB3 ROland" + str(x) + ".png")
    new = Image.open("C:\\Users\\sergi\\PycharmProjects\\AILAB3 ROland" + str(x) + ".png")
    new.show()

    image = Image.open(image_path)

    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    factor = 1.5  # adjust this factor as needed
    enhanced_image = enhancer.enhance(factor)

    # Sharpen the enhanced image
    sharpened_image = enhanced_image.filter(ImageFilter.SHARPEN)

    grayscale_image = sharpened_image.convert('L')

    # Resize the grayscale image to a bigger size
    resized_image = grayscale_image.resize((1200,1200), Image.LANCZOS)
    # Save the sharpened image
    sharpened_image_path = "sharpened_image.jpg"
    resized_image.save(sharpened_image_path)
    resized_image.show()


    EX1("C:\\Users\\sergi\\PycharmProjects\\AILAB3 ROland" + str(x) + ".png", text, client)
    EX1(sharpened_image_path, text, client)


def main():
    image_path1 = "C:\\Users\\sergi\\PycharmProjects\\AILAB3 ROland\\test1.png"
    image_path2 = "C:\\Users\\sergi\\PycharmProjects\\AILAB3 ROland\\test2.jpeg"
    border_path1 = "C:\\Users\\sergi\\PycharmProjects\\AILAB3 ROland\\borders1.txt"
    border_path2 = "C:\\Users\\sergi\\PycharmProjects\\AILAB3 ROland\\borders2.txt"

    #rezultatele procesarii imaginilor
    result = process_image(image_path1, computervision_client)
    #print(result)
    result = process_image(image_path2, computervision_client)
    #print(result)

    text1 = "Google Cloud Platform"
    text2 = "Succes in rezolvarea tEMELOR la LABORAtoarele de Inteligenta Artificiala!"
    #print("REZOLVAREA PRIMEI IMAGINI")
    #EX1(image_path1, text1, computervision_client)
    #print("REZOLVARE IMAGINEA 2")
    #EX1(image_path2, text2, computervision_client)

    #ttxt = EX2(image_path1, border_path1, computervision_client)
    #print(ttxt)
    #EX2(image_path2, border_path2, computervision_client)


    #EX3(image_path1, text1, computervision_client)
    #EX3(image_path2, text2, computervision_client)

main()
