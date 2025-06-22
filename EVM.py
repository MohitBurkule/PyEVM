import cv2
import numpy as np
import scipy.signal as signal
import scipy.fftpack as fftpack
from pympler import asizeof
import sys
import argparse
import os
import tempfile
import subprocess

def get_total_size_in_mb(obj):
    size_in_bytes = asizeof.asizeof(obj)
    size_in_mb = size_in_bytes / (1024 * 1024)  # Convert bytes to MB
    return round(size_in_mb, 2)

def compress_video_ffmpeg(input_path, max_dimension=1000):
    """
    Compress video using FFmpeg by resizing it to have maximum dimension of max_dimension pixels
    while maintaining aspect ratio. Returns path to temporary compressed video.
    """
    # Create temporary file for compressed video
    temp_fd, temp_path = tempfile.mkstemp(suffix='.mp4')
    os.close(temp_fd)  # Close file descriptor
    
    # Get original video properties using OpenCV
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Cannot open video file: ", input_path)
    
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # Calculate new dimensions maintaining aspect ratio
    if original_width > original_height:
        if original_width > max_dimension:
            new_width = max_dimension
            new_height = int((original_height * max_dimension) / original_width)
        else:
            new_width = original_width
            new_height = original_height
    else:
        if original_height > max_dimension:
            new_height = max_dimension
            new_width = int((original_width * max_dimension) / original_height)
        else:
            new_width = original_width
            new_height = original_height
    
    # Ensure dimensions are even (required by most codecs)
    new_width = new_width - (new_width % 2)
    new_height = new_height - (new_height % 2)
    
    print("Compressing video:", original_width, original_height, "->", new_width, new_height)
    
    # Use FFmpeg to compress
    cmd = [
        'ffmpeg', '-i', input_path,
        '-vf', 'scale={}:{}:flags=lanczos'.format(new_width, new_height),
        '-c:v', 'libx264',
        '-crf', '23',  # Good quality/size balance
        '-preset', 'medium',
        '-y',  # Overwrite output file
        temp_path
    ]
    
    try:
        # Use subprocess.check_call for Python 3.5 compatibility
        with open(os.devnull, 'w') as devnull:
            subprocess.check_call(cmd, stdout=devnull, stderr=devnull)
        print("Compressed video saved temporarily to:", temp_path)
        return temp_path
    except subprocess.CalledProcessError as e:
        print("FFmpeg compression failed:", str(e))
        print("Falling back to OpenCV compression...")
        # Remove the failed temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        # Fall back to OpenCV method
        return compress_video_opencv(input_path, max_dimension)

def compress_video_opencv(input_path, max_dimension=1000):
    """
    Fallback: Compress video using OpenCV by resizing it to have maximum dimension of max_dimension pixels
    while maintaining aspect ratio. Returns path to temporary compressed video.
    """
    # Create temporary file for compressed video
    temp_fd, temp_path = tempfile.mkstemp(suffix='.mp4')
    os.close(temp_fd)  # Close file descriptor as we'll use OpenCV to write
    
    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Cannot open video file: ", input_path)
    
    # Get original video properties
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate new dimensions maintaining aspect ratio
    if original_width > original_height:
        if original_width > max_dimension:
            new_width = max_dimension
            new_height = int((original_height * max_dimension) / original_width)
        else:
            new_width = original_width
            new_height = original_height
    else:
        if original_height > max_dimension:
            new_height = max_dimension
            new_width = int((original_width * max_dimension) / original_height)
        else:
            new_width = original_width
            new_height = original_height
    
    # Ensure dimensions are even (required by some codecs)
    new_width = new_width - (new_width % 2)
    new_height = new_height - (new_height % 2)
    
    print("Compressing video (OpenCV fallback):", original_width, original_height, "->", new_width, new_height)
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_path, fourcc, fps, (new_width, new_height))
    
    # Process frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        out.write(resized_frame)
    
    # Release resources
    cap.release()
    out.release()
    
    print("Compressed video saved temporarily to:", temp_path)
    return temp_path

def compress_video(input_path, max_dimension=1000, use_ffmpeg=True):
    """
    Compress video by resizing it to have maximum dimension of max_dimension pixels
    while maintaining aspect ratio. Returns path to temporary compressed video.
    Uses FFmpeg by default, falls back to OpenCV if FFmpeg fails.
    """
    if use_ffmpeg:
        return compress_video_ffmpeg(input_path, max_dimension)
    else:
        return compress_video_opencv(input_path, max_dimension)

#convert RBG to YIQ
def rgb2ntsc(src):
    [rows,cols]=src.shape[:2]
    dst=np.zeros((rows,cols,3),dtype=np.float64)
    T = np.array([[0.114, 0.587, 0.298], [-0.321, -0.275, 0.596], [0.311, -0.528, 0.212]])
    for i in range(rows):
        for j in range(cols):
            dst[i, j]=np.dot(T,src[i,j])
    return dst

#convert YIQ to RBG
def ntsc2rbg(src):
    [rows, cols] = src.shape[:2]
    dst=np.zeros((rows,cols,3),dtype=np.float64)
    T = np.array([[1, -1.108, 1.705], [1, -0.272, -0.647], [1, 0.956, 0.620]])
    for i in range(rows):
        for j in range(cols):
            dst[i, j]=np.dot(T,src[i,j])
    return dst

#Build Gaussian Pyramid
def build_gaussian_pyramid(src,level=3):
    s=src.copy()
    pyramid=[s]
    for i in range(level):
        s=cv2.pyrDown(s)
        pyramid.append(s)
    return pyramid

#Build Laplacian Pyramid
def build_laplacian_pyramid(src,levels=3):
    gaussianPyramid = build_gaussian_pyramid(src, levels)
    pyramid=[]
    for i in range(levels,0,-1):
        GE=cv2.pyrUp(gaussianPyramid[i])
        # Resize GE to match the exact size of gaussianPyramid[i-1]
        h, w = gaussianPyramid[i-1].shape[:2]
        GE = cv2.resize(GE, (w, h))
        L=cv2.subtract(gaussianPyramid[i-1],GE)
        pyramid.append(L)
    return pyramid

#load video from file
def load_video(video_filename):
    cap=cv2.VideoCapture(video_filename)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    video_tensor=np.zeros((frame_count,height,width,3),dtype='float')
    x=0
    while cap.isOpened():
        ret,frame=cap.read()
        if ret is True:
            video_tensor[x]=frame
            x+=1
        else:
            break
    return video_tensor,fps

# apply temporal ideal bandpass filter to gaussian video
def temporal_ideal_filter(tensor,low,high,fps,axis=0):
    fft=fftpack.fft(tensor,axis=axis)
    frequencies = fftpack.fftfreq(tensor.shape[0], d=1.0 / fps)
    bound_low = (np.abs(frequencies - low)).argmin()
    bound_high = (np.abs(frequencies - high)).argmin()
    fft[:bound_low] = 0
    fft[bound_high:-bound_high] = 0
    fft[-bound_low:] = 0
    iff=fftpack.ifft(fft, axis=axis)
    return np.abs(iff)

# build gaussian pyramid for video
def gaussian_video(video_tensor,levels=3):
    for i in range(0,video_tensor.shape[0]):
        frame=video_tensor[i]
        pyr=build_gaussian_pyramid(frame,level=levels)
        gaussian_frame=pyr[-1]
        if i==0:
            vid_data=np.zeros((video_tensor.shape[0],gaussian_frame.shape[0],gaussian_frame.shape[1],3))
        vid_data[i]=gaussian_frame
    return vid_data

#amplify the video
def amplify_video(gaussian_vid,amplification=50):
    return gaussian_vid*amplification

#reconstract video from original video and gaussian video
def reconstract_video(amp_video,origin_video,levels=3):
    final_video=np.zeros(origin_video.shape)
    for i in range(0,amp_video.shape[0]):
        img = amp_video[i]
        for x in range(levels):
            img=cv2.pyrUp(img)
        img=img+origin_video[i]
        final_video[i]=img
    return final_video

#save video to files
def save_video(video_tensor, output_path="/app/output/out.avi"):
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    [height,width]=video_tensor[0].shape[0:2]
    writer = cv2.VideoWriter(output_path, fourcc, 30, (width, height), 1)
    for i in range(0,video_tensor.shape[0]):
        writer.write(cv2.convertScaleAbs(video_tensor[i]))
    writer.release()
    print("Video saved to: ",output_path)

#magnify color
def magnify_color(video_name,low,high,levels=3,amplification=20,output_path="/app/output/out.avi",compress=True,max_dimension=1000, use_ffmpeg=True):
    # Compress video if enabled
    if compress:
        compressed_video_path = compress_video(video_name, max_dimension, use_ffmpeg)
        video_to_process = compressed_video_path
    else:
        video_to_process = video_name
    
    try:
        t,f=load_video(video_to_process)
        gau_video=gaussian_video(t,levels=levels)
        filtered_tensor=temporal_ideal_filter(gau_video,low,high,f)
        amplified_video=amplify_video(filtered_tensor,amplification=amplification)
        final=reconstract_video(amplified_video,t,levels=3)
        save_video(final, output_path)
    finally:
        # Clean up temporary compressed video
        if compress and os.path.exists(compressed_video_path):
            os.unlink(compressed_video_path)
            print("Temporary compressed video deleted: ",compressed_video_path)

#build laplacian pyramid for video
def laplacian_video(video_tensor,levels=3):
    tensor_list=[]
    for i in range(0,video_tensor.shape[0]):
        frame=video_tensor[i]
        pyr=build_laplacian_pyramid(frame,levels=levels)
        if i==0:
            for k in range(levels):
                tensor_list.append(np.zeros((video_tensor.shape[0],pyr[k].shape[0],pyr[k].shape[1],3)))
        for n in range(levels):
            tensor_list[n][i] = pyr[n]
    return tensor_list

#butterworth bandpass filter
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    omega = 0.5 * fs
    low = lowcut / omega
    high = highcut / omega
    b, a = signal.butter(order, [low, high], btype='band')
    y = signal.lfilter(b, a, data, axis=0)
    return y

#reconstract video from laplacian pyramid
def reconstract_from_tensorlist(filter_tensor_list,levels=3):
    final=np.zeros(filter_tensor_list[-1].shape)
    for i in range(filter_tensor_list[0].shape[0]):
        up = filter_tensor_list[0][i]
        for n in range(levels-1):
            up=cv2.pyrUp(up)+filter_tensor_list[n + 1][i]#可以改为up=cv2.pyrUp(up)
        final[i]=up
    return final

#manify motion
def magnify_motion(video_name,low,high,levels=3,amplification=20,output_path="/app/output/out.avi",compress=True,max_dimension=1000, use_ffmpeg=True):
    # Compress video if enabled
    if compress:
        compressed_video_path = compress_video(video_name, max_dimension, use_ffmpeg)
        video_to_process = compressed_video_path
    else:
        video_to_process = video_name
    
    try:
        t,f=load_video(video_to_process)
        lap_video_list=laplacian_video(t,levels=levels)#del t after this
        filter_tensor_list=[]#12.5 to 4.5 here
        del t#8 here
        for i in range(levels):
            filter_tensor=butter_bandpass_filter(lap_video_list[i],low,high,f)
            filter_tensor*=amplification
            filter_tensor_list.append(filter_tensor)
            lap_video_list[i]=None#min goes to 4.5 again but after this 8.2 available
        recon=reconstract_from_tensorlist(filter_tensor_list)
        del filter_tensor_list
        del lap_video_list
        del filter_tensor
        t,f=load_video(video_to_process)# del f again possible
        del f
        final=t+recon#again 4.5 here
        del t
        del recon
        save_video(final, output_path)
    finally:
        # Clean up temporary compressed video
        if compress and os.path.exists(compressed_video_path):
            os.unlink(compressed_video_path)
            print("Temporary compressed video deleted:",compressed_video_path)

def fix_aspect_ratio_with_ffmpeg(input_path, output_path, target_width, target_height):
    """Use FFmpeg to fix aspect ratio issues"""
    cmd = [
        'ffmpeg', '-i', input_path,
        '-vf', 'scale={}:{}:flags=lanczos'.format(target_width, target_height),
        '-aspect', '{}:{}'.format(target_width, target_height),
        '-c:v', 'libx264',
        '-crf', '18',
        '-y',  # Overwrite output file
        output_path
    ]
    
    try:
        with open(os.devnull, 'w') as devnull:
            subprocess.check_call(cmd, stdout=devnull, stderr=devnull)
        print("Aspect ratio corrected:", output_path)
        return True
    except subprocess.CalledProcessError as e:
        print("FFmpeg error:", str(e))
        return False

def debug_frame_shape(video_path):
    """Debug function to check frame shapes throughout processing"""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if ret:
        print("Original frame shape: ",frame.shape)
        print("Frame aspect ratio: ",frame.shape[1]/frame.shape[0])
    cap.release()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Eulerian Video Magnification')
    parser.add_argument('video_path', help='Path to the MP4 video file')
    parser.add_argument('--mode', choices=['color', 'motion'], default='motion',
                        help='Magnification mode: color or motion (default: motion)')
    parser.add_argument('--low', type=float, default=0.4,
                        help='Low frequency bound (default: 0.4)')
    parser.add_argument('--high', type=float, default=3.0,
                        help='High frequency bound (default: 3.0)')
    parser.add_argument('--levels', type=int, default=3,
                        help='Number of pyramid levels (default: 3)')
    parser.add_argument('--amplification', type=int, default=20,
                        help='Amplification factor (default: 20)')
    parser.add_argument('--output', type=str, default="/app/output/out.avi",
                        help='Output video path (default: /app/output/out.avi)')
    parser.add_argument('--no-compress', action='store_true',
                        help='Disable video compression (enabled by default)')
    parser.add_argument('--max-dimension', type=int, default=960,
                        help='Maximum dimension for video compression (default: 960)')
    parser.add_argument('--use-opencv', action='store_true',
                        help='Use OpenCV for compression instead of FFmpeg (FFmpeg is default)')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Compression is enabled by default, disabled only if --no-compress is specified
    compress_enabled = not args.no_compress
    use_ffmpeg = not args.use_opencv  # FFmpeg is default unless --use-opencv is specified

    if args.max_dimension != 960:
        print("Warning: Using a max-dimension different from 960 pixels may affect the processing results")

    if args.mode == 'color':
        magnify_color(args.video_path, args.low, args.high, args.levels, args.amplification, args.output, compress_enabled, args.max_dimension, use_ffmpeg)
    else:
        magnify_motion(args.video_path, args.low, args.high, args.levels, args.amplification, args.output, compress_enabled, args.max_dimension, use_ffmpeg)