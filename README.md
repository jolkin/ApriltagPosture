# Apriltag Tracker

The following is code that will retrieve planar coordinates of a camera using an apriltag.

## Dependencies

This code requires the apriltag module, numpy and OpenCV 3.4.1


## Figuring Out The Experimental Factor

As described in the documentation for the init function for the ApriltagTracker class, you will need to
calculate an experimental constant to scale coordinates calculated by the object into real-world units. This
will vary from camera-to-camera, but can be easily found by following these steps:

1. Place the camera directly in front of the apriltag.
2. Set "exp_factor" in the code to 1.
3. Run the code, and record both the distance the camera is from the apriltag as well as the calculated y coordinate
(the x coordinate should be approximately zero)
4. Repeat this for a number of different distances, but remain directly in front of the tag.
5. Calculate the ratio between each distance and its correspond y coordinate. 
6. Set "exp_factor" to the average of these ratios.

