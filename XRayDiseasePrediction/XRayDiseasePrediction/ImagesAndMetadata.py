import Utils
import sys
import numpy as np
class ImagesAndMetadata(object):
	def __init__(self,metadata, labels):
		self._metadata = metadata
		self._labels = labels
		self._num_images = len(metadata)
		self._epochs_completed = 0
		self._index_in_epoch = 0

	
	@property
	def labels(self):
		return self._labels


	@property
	def metadata(self):
		return self._metadata
	
	@property
	def num_images(self):
		return self._num_images

	@property
	def epochs_completed(self):
		return self._epochs_completed

	def next_batch(self, batch_size, img_dir_name):
		"""Return the next `batch_size` examples from this data set."""
		#print("Inside next batch "+img_dir_name)
		start = self._index_in_epoch
		self._index_in_epoch += batch_size
		if self._index_in_epoch > self._num_images:
			# Finished epoch
			self._epochs_completed += 1
			start = 0
			self._index_in_epoch = batch_size
		end = self._index_in_epoch
		#print("start="+str(start)+" end ="+str(end)+" _metadata shape="+str(self._metadata.shape))
		image_names = self._metadata[start: end,3].tolist()
		first = True
		for x in image_names:
			try:
				#print("Creating vector for "+":"+str(x))
				image_np = Utils.CreateVector(img_dir_name, x)
				#print(str(image_np.shape))
				if(first):
					image_array = np.expand_dims(image_np, axis=0)
					#print(str(image_array.shape))
					first = False
				else:
					new_image_array = np.expand_dims(image_np, axis=0)
					image_array = np.append(image_array, new_image_array, axis=0)
				#print(image_array.shape)
			except Exception:
				print("File not found")
		return image_array, self._metadata[start:end, 0:3], self._labels[start:end]

if __name__ == "__main__":
	args = sys.argv
	img_dir_name = args[1]
	debug_log_file = args[3]
	csv_file = args[2]

	debug_logs = open(debug_log_file, 'w')
	meta_data, labels = Utils.load_csv_file_without_images( csv_file, debug_logs)
	imagesAndMetadata = ImagesAndMetadata(meta_data, labels)
	_, meta_1, label_1=imagesAndMetadata.next_batch(2, img_dir_name)
	print(meta_1.tolist())
	print("Next batch")
	_, meta_1, label_1=imagesAndMetadata.next_batch(2, img_dir_name)
	print(meta_1.tolist())
	debug_logs.close()
	#CreateVector(img_dir_name, csv_file, debug_log_file)
	print("Enter any char...")
	name = sys.stdin.readline()
	#"C:\\Users\\rajgup\\Documents\\pythonAnaconda\\XRayDiseasePrediction\\sampleImages" "C:\\Users\\rajgup\\Documents\\pythonAnaconda\\XRayDiseasePrediction\\train_sample2.csv" "C:\\Users\\rajgup\\Documents\\pythonAnaconda\\XRayDiseasePrediction\\log1.txt"
