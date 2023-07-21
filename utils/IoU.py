import numpy as np

def calculate_iou(pred_mask, label_mask):
    intersection = np.logical_and(pred_mask, label_mask)
    union = np.logical_or(pred_mask, label_mask)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def calculate_accuracy(pred_mask, label_mask):
    correct_pixels = np.sum(pred_mask == label_mask)
    total_pixels = pred_mask.size
    accuracy = correct_pixels / total_pixels
    return accuracy

def compute_metrics(predictions, labels, seg_classes):
    class_iou = {}
    class_accuracy = {}
    confusion_matrix = np.zeros((len(seg_classes['Scenes']), len(seg_classes['Scenes'])), dtype=np.int32)
    pred_mask_temp = np.empty((0, predictions.shape[1]), dtype=np.bool)
    label_mask_temp = np.empty((0, labels.shape[1]), dtype=np.bool)
    for class_id in seg_classes['Scenes']:
        pred_mask = (predictions == class_id)
        label_mask = (labels == class_id)

        iou = calculate_iou(pred_mask, label_mask)
        accuracy = calculate_accuracy(pred_mask, label_mask)

        class_iou[class_id] = iou
        class_accuracy[class_id] = accuracy

        # Update confusion matrix
        for i in range(len(seg_classes['Scenes'])):
            confusion_matrix[i][class_id] = np.sum(np.logical_and(predictions == class_id, labels == i))

        pred_mask_temp = np.concatenate((pred_mask_temp, pred_mask), axis=0)
        label_mask_temp = np.concatenate((label_mask_temp, label_mask), axis=0)
        if class_id < 5:
            pred_mask_temp_without_static = pred_mask_temp
            label_mask_temp_without_static = label_mask_temp

    total_iou = calculate_iou(pred_mask_temp, label_mask_temp)
    total_accuracy = calculate_accuracy(pred_mask_temp, label_mask_temp)
    total_iou_without_static = calculate_iou(pred_mask_temp_without_static, label_mask_temp_without_static)
    total_accuracy_without_static = calculate_accuracy(pred_mask_temp_without_static, label_mask_temp_without_static)

    print("Class IoU:")
    for class_id, iou in class_iou.items():
        print(f"Class {class_id}: {iou:.4f}")

    print("Class Accuracy:")
    for class_id, accuracy in class_accuracy.items():
        print(f"Class {class_id}: {accuracy:.4f}")

    print("Total IoU:", total_iou)
    print("Total Accuracy:", total_accuracy)
    
    print("Total IoU without static:", total_iou_without_static)
    print("Total Accuracy without static:", total_accuracy_without_static)

    print("Confusion Matrix:")
    print(confusion_matrix)

    print("Confusion Matrix (percentages):")
    confusion_matrix = confusion_matrix.astype(np.float32)

    for i in range(len(seg_classes['Scenes'])):
        confusion_matrix[i] /= np.sum(confusion_matrix[i])
    confusion_matrix *= 100
    # keep the decimal place to one place
    confusion_matrix = np.round(confusion_matrix, 1)
    print(confusion_matrix)

    return total_iou, total_accuracy
    
