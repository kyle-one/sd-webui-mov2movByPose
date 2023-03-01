import launch

print('Installing requirements for Mov2movByPose')
if not launch.is_installed("cv2"):
    launch.run_pip("install opencv-python", "requirements for opencv")
