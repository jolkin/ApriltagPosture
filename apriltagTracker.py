"""
Implementation of pose recovery using apriltags without the use of ROS
Extracts planar coordinates from apriltag homography
All recorded points are logged in plotPos.txt as x,y coordinates
"""


import apriltag #If the apriltag module is imported normally, replace this
import numpy as np
import cv2
import sys
import math
import os
import threading as threads
import queue

q = queue.LifoQueue(maxsize=200)

os.system('v4l2-ctl -d /dev/video1 -c exposure_absolute=20 -c gain=255 -c exposure_auto=1 -c backlight_compensation=1 -c brightness=128')

# Set to width and height of the camera's image in pixels
width = 960
height = 720

# experimental scaling factor
exp_factor = 4.4526

# calibrated camera matrix - will vary from camera to camera
cameraMatrix = np.array([[958.681002, 0, 446.220075],
                [0, 956.629787, 368.525809],
                [0, 0, 1]])


#projectionMatrix = np.array([[914.75238, 0, 641.295374],
#                             [0, 926.169067, 388.841062],
#                             [0, 0, 1]])

# exp mtx
bigMtx = np.array([[902.807079, 0, 636.978099],
                [0, 908.223764, 387.758118],
                [0, 0, 1]])
exp_distortion = np.array([0.106361, -0.161914, 0.001708, 0.003198, 0])


# edge length of the apriltag in meters
cornerDist = (0.245 - 0.075) / 2

# calibrated camera distortion - will vary from camera to camera
distortion = np.array([0.116232, -0.197929, 0.002765, 0.002874, 0])



# optimal camera matrix used to restorting distorted images


class ApriltagTracker:

    def __init__(self, w, h, cameraMatrix, distortion, cornerDist, video,
                 tag="0", exp_factor=4.4526):
        """
        Initialize the tracker object
        :param w: width of the image in pixels
        :param h: height of the image in pixels
        :param cameraMatrix: intrinsic camera matrix
        :param distortion: distortion coefficients
        :param cornerDist: distance along the edge of the april tag, divided by 2 (in cm)
        :param cameraID: ID number of the camera
        :param tag: unique identifier to be applied to all the output files
        :param exp_factor: experimental factor (see ReadMe)
        """
        self.width = w
        self.height = h
        self.cameraMatrix = cameraMatrix
        self.distortion = distortion
        self.cornerDist = cornerDist
        self.exp_factor = exp_factor
        newMtrx, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distortion, (width, height), 1, (width, height))
        self.fixedMatrix = newMtrx
        self.camera = video
        # initialize x and y coord so we don't get errors later
        self.xCoord = 0
        self.yCoord = 0
        self.zCoord = 0

        # initialize array for frames to be stored in
        self.frames = []

        # initialize angle relative to apriltag (also just so we don't get errors later)
        self.phi = 0
        self.rho = 0
        self.tag = tag

        self.pixelsFromCenterX = 0

        self.pixelsFromCenterY = 0

        # initialize apriltag detector
        self.detector=apriltag.Detector()

        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        self.videoWriter = cv2.VideoWriter('cameraOutput'+self.tag+'.avi',fourcc, 20.0, (960, 720))

    def getFeedback(self):
        return self.feedback

    def analyzeFrame(self, image=None):
        """
        pops the latest frame off the stack of frames created by takePics and determines the x coord, y coord,
        and orientation of the camera relative to the april tag in the frame. If there is no apriltag in the frame it
        return whatever was stored here from the last call

        This also will return a measurement of the distance between the center of the frame, and the center of the
        apriltag (called the "frame offset"). This is a measurement from -1 to 1 that is proportional to the distance
        in pixels between the detected center of the apriltag and the center of the frame.

        :param image optional argument containing the picture you wish to analyze. Only works when the queue is empty
        :return: x, y, z, yaw, pitch horizontal frame offset, vertical frame offset
        """
        if not q.empty() or image is not None:
            frame = image
            if not q.empty():
                frame = q.get(False)
            frame = cv2.undistort(frame, self.cameraMatrix, self.distortion, None, self.fixedMatrix)
            # convert to gray for apriltag detection
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            detections, img = self.detector.detect(gray, return_image=True)


            # iterate through all detections, if there are no tags detected this will be skipped
            for i, detection in enumerate(detections):
                # calculate position and distance
                dist, pos, zPos = self.homographyDist(detection, self.fixedMatrix)
                origin = np.zeros(3)

                self.pixelsFromCenterX = (width / 2 - detection.center[0]) / (width / 2)
                self.pixelsFromCenterY = (height / 2 - detection.center[1]) / (height / 2)

                # calculate the planar coordinates
                x, y, z, theta, self.phi, self.rho = self.toCoords(detection, pos, dist, self.fixedMatrix)
                l = self.isLeft(pos, detection.corners)
                x = x * l
                if self.rho < 0:
                    self.rho = self.rho+math.pi*2
                calculatedTheta = (theta-math.pi/2)


                # reproject onto image overlay
                cv2.line(frame, (int(len(img[0]) / 2), 0), (int(len(img[0]) / 2), len(img)), (0, 0, 255))
                cv2.line(frame, (0, int(len(img) / 2)), (len(img[0]), int(len(img) / 2)), (0, 0, 255))
                cv2.circle(frame, (int(detection.center[0]), int(detection.center[1])), 5, (0, 255, 0))
                projPoints, jacobian = cv2.projectPoints(np.array([pos]), origin, origin, self.fixedMatrix, self.distortion)
                cv2.circle(frame, (int(projPoints[0][0][0]), int(projPoints[0][0][1])), 5, (0, 0, 255))
                outputInfo = "x: {}, y: {}".format(x, y)
                cv2.putText(frame, outputInfo, (50,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

                # set local vals and write frame out to video
                self.xCoord = x
                self.yCoord = y
                self.zCoord = z
                self.videoWriter.write(frame)
            cv2.imshow("live", frame)
            cv2.waitKey(1)
        return self.xCoord, self.yCoord, self.zCoord, self.phi, self.rho, self.pixelsFromCenterX, self.pixelsFromCenterY

    def toCoords(self, detection, pos, dist, cameraMatrix):
        """
        Calculates the planar coordinates from calculated distance and homography
        :param detection: apriltag detection object
        :param pos: translation vector of the robot
        :param dist: calculated distance from apriltag
        :param cameraMatrix: intrinsic camera matrix
        :return: x,y planar coordinates as a top-down view with 0,0 at the center of the apriltag
        """

        # retrieve rotation vectors from homography
        num, rotvec, tvec, normvec = cv2.decomposeHomographyMat(detection.homography, cameraMatrix)

        # in experimentation solution 2 always yielded the correct solution
        optI = 3
        goalrotvec = rotvec[3]

        fourdmtx = np.matrix([[goalrotvec[0, 0], goalrotvec[0, 1], goalrotvec[0, 2], pos[0]],
                              [goalrotvec[1, 0], goalrotvec[1, 1], goalrotvec[1, 2], pos[1]],
                              [goalrotvec[2, 0], goalrotvec[2, 1], goalrotvec[2, 2], pos[2]],
                              [0., 0., 0., 1.]])
        # decompose rotation matrix into euler angles
        vals = decomposeSO3(rotvec[optI])
        inverted = -np.linalg.inv(fourdmtx) * np.sign(1 - abs(vals[2]))


        theta = (vals[1]*np.sign(1-abs(vals[2]))) - (math.atan2((pos[0]), pos[2]))
        subtract =  -vals[0]
        if np.sign(1-abs(vals[2])) > 0:
            subtract = -(vals[0] + math.pi)

        phi = subtract - (math.atan2((pos[1]), pos[2]))

        # calculate planar coordinates as a function of the distance and the y rotation
        x = abs(dist * math.sin(theta))
        y = dist * math.cos(theta)
        z = dist * math.sin(phi)

        return inverted[0], inverted[2], inverted[1], theta, (vals[1]*np.sign(1-vals[2])), subtract

    def getCoords(self):
        """
        get the coordinates and simple PID feedback on gaze
        :return: the x position in cm, y position in cm, and feedback
        """

        return self.xCoord, self.yCoord, self.feedback

    def isLeft(self, pos, corners):
        """
        Determines if the robot is to the left or to the right of the apriltag
        from the perspective of facing the apriltag
        :param pos: translation vector of the robot (generated from homography)
        :param corners: position of the corners of the apriltag in image space
        :return: -1 if the robot is to the left, 1 if the robot is to the right
        """


        d1 = np.array([pos[0] - self.cornerDist, pos[1] + self.cornerDist, pos[2]])
        d2 = np.array([pos[0] + self.cornerDist, pos[1] + self.cornerDist, pos[2]])
        d3 = np.array([pos[0] + self.cornerDist, pos[1] - self.cornerDist, pos[2]])
        d4 = np.array([pos[0] - self.cornerDist, pos[1] - self.cornerDist, pos[2]])

        testRatio = (np.linalg.norm(d1)+np.linalg.norm((d4)))/(np.linalg.norm(d2)+np.linalg.norm((d3)))

        measuredRatio = (corners[3][1] - corners[0][1])/(corners[2][1]-corners[1][1])

        if testRatio > measuredRatio:
            return 1
        return -1


    def homographyDist(self, detection, mtx):
        """
        Computes the position of the camera, and the distance to the camera

        :param detection: apriltag detection object
        :param mtx: intrinsic camera matrix
        :return: the distance to the apriltag, and the position
        """

        # create extentrics matrix without accounting for s
        scaledExtrinsics = np.zeros((3, 3))
        for x in range(0, 2):
            scaledExtrinsics[x] = (detection.homography[x]-detection.homography[2]*mtx[x][2])/mtx[x][x]
        scaledExtrinsics[2] = detection.homography[2]

        # calculate s as the geometric mean of the magnitudes of the first two columns
        magnitudes = np.linalg.norm(scaledExtrinsics, axis=0)
        s = np.sqrt(magnitudes[0]+magnitudes[1])

        # ensure z is positive
        if detection.homography[2][2] < 0:
            s = s *-1

        scaledExtrinsics = scaledExtrinsics/s


        return scaledExtrinsics[2][2] * self.exp_factor, self.exp_factor*scaledExtrinsics[:3,2], scaledExtrinsics[1][2]*self.exp_factor


    def startTracking(self):
        """
        Start multiple threads to perform tracking
        :return: None
        """
        thread = threads.Thread(target=takePics, args=(self.camera, self.width, self.height))
        thread.start()
        while True:
            x, y, z, phi, rho, hOffset, vOffset = self.analyzeFrame()
            print("{},{},{} | {},{}".format(x, y, z, phi, rho))

def takePics(camera, width, height):
    """
    Take pictures and add them to the queue of picture to be processed
    :return: none
    """
    video = cv2.VideoCapture(camera)
    video.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    video.set(cv2.CAP_PROP_FPS, 30)



    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    ok, frame = video.read()
    if not ok:
        print("Cannot read video")
        sys.exit()

    while True:
        ok, frame = video.read()
        q.put(frame)



def decomposeSO3(rotationMatrix):
    thetaX = math.atan2(rotationMatrix[2, 1], rotationMatrix[2, 2])
    thetaY = math.atan2(-rotationMatrix[2, 0], math.hypot(rotationMatrix[2, 1], rotationMatrix[2, 2]))
    thetaZ = math.atan2(rotationMatrix[1, 0], rotationMatrix[0, 0])
    return np.array((thetaX, thetaY, thetaZ))


if __name__ == "__main__":
    tracker = ApriltagTracker(960, 720, cameraMatrix, distortion, cornerDist, 0)
    tracker.startTracking()