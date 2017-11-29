def check_gpu(verbose=1):
	import os

	if verbose:
		try:
			print('\n =========== NVIDIA =========== ')
			print('0. Nvidia Hardware (lspci | grep -i nvidia) : ')
			tmp = [each for each in os.popen('lspci | grep -i nvidia')]
			if len(tmp): print(' ---> ', tmp[0])
			else: print(' ---> No Graphics Card Found')

			tmp = [each for each in os.popen('cat /proc/driver/nvidia/version')]
			print('0. Nvidia Driver (cat /proc/driver/nvidia/version) : ')
			if len(tmp): print(' ---> ', tmp[0])
			else: print(' ---> No Nvidia driver found ...')

			print('0. Nvidia Driver (ls /usr/lib | grep nvidia-) : ')
			tmp = [each.strip('\n') for each in os.popen('ls /usr/lib | grep nvidia-')]
			if len(tmp): print(' ---> ', tmp)
			else: print(' ---> No Nvidia driver found ...')

			print('\n0. Nvidia Packages : (dpkg --get-selections | grep nvidia) : ')
			tmp = [each for each in os.popen('dpkg --get-selections | grep nvidia')]
			if len(tmp):
				for each in os.popen('dpkg --get-selections | grep nvidia'):
					print(' ---> ', each.strip('\n'))
			else:
				print(' ---> No Nvidia packages found ...')

			print('\n0. Nvidia Control Panel : nvidia-smi')
			tmp = [each.strip('\n').strip('|') for each in os.popen('nvidia-smi')]
			if len(tmp):
				print(' ---> ', tmp[2])
			else:
				print(' ---> No Nvidia Control Panel Detected')

			print('\n =========== CUDA =========== ')
			print('1. Cuda Version : nvcc -V')
			tmp = [each.strip('\n') for each in os.popen('nvcc -V')]
			if len(tmp): print(' ---> ', tmp[-1])
			try:
				print('1. Envs  : PATH (containing cuda)', [each for each in os.environ['PATH'].split(':') if each.find('cuda') > -1])
				print('1. Envs  : CUDA_HOME', os.environ['CUDA_HOME'])
				print('1. Envs  : CUDA_ROOT', os.environ['CUDA_ROOT'])
				print('1. Envs  : LD_LIBRARY_PATH:', os.environ['LD_LIBRARY_PATH'])
				tmp_cmd = 'find ' + os.environ['CUDA_HOME'] + ' -name *dnn*'
				print('1. CUDnn :', [ each.replace('\n', '') for each in os.popen(tmp_cmd)])
			except:
				print ('Error : COuld not find local CUDA envs')
				
			print('\n =========== CUDA & NVIDIA ========== ')
			try:
				tmp_path = os.environ['CUDA_HOME'] + '/samples/1_Utilities/deviceQuery/deviceQuery'
				print('2. ', tmp_path)
				tmp = [each.strip('\n') for each in os.popen(tmp_path)]
				if len(tmp):
					print(' ---> ', tmp[4])
					print(' ---> ', tmp[6])
					print(' ---> ', tmp[-1])
					print(' --->  CUDA successfully detected Nvidia drivers')
				else:
					print(' ---> CUDA did not detect Nvidia drivers')
			except:
				print (' ---> Some error')

		except:
			print('Error:')
			
	print('\n =========== nvidia-smi ========== ')
	tmp = [each.strip('\n') for each in os.popen('nvidia-smi --query-compute-apps=pid --format=csv,noheader')]
	print ('Query for existing PIDs using GPU : nvidia-smi --query-compute-apps=pid --format=csv,noheader')
	print (' ---> ', tmp)

	if len(tmp):
		print ('Damn son! You gotta kill the PIDS - {0} and then run nvidia-smi -r under root'.format(tmp))
		print (' ---> Then come back and run this script again')
	else:
		import tensorflow as tf    
		from tensorflow.python.client import device_lib
		devices = device_lib.list_local_devices()
		for device in devices:
			print ('TensorFlow Devices:', str(device.name).replace('\n',''))

		if len(devices) > 1:
			print ('\n')
			return 1  
		else:
			print (' --> Try resetting the Nvidia GPU')
			print ('\n')
			return 0