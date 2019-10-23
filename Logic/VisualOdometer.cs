using Emgu.CV;
using Emgu.CV.Structure;
using System;
using System.Collections.Generic;

namespace Egomotion
{
    public struct OdometerFrame
    {
        public TimeSpan TimeDiff { get; set; }
        public Mat Translation { get; set; }
        public Mat Velocity { get; set; }
        public Mat Rotation { get; set; } // Matrix or euler angles ?
        public Mat AngularVelocity { get; set; }
    }

    public interface IVisualOdometer
    {
        OdometerFrame ComputeOdometry(Mat frame1, Mat frame2);

        // Draws results of last computations on given image (it should be same as frame2 above)
        void Visualize(Image<Bgr, byte> image);
    }
}
