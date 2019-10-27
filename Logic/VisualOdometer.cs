using Emgu.CV;
using Emgu.CV.Structure;
using System;
using System.Collections.Generic;

namespace Egomotion
{
    public struct OdometerFrame
    {
        public TimeSpan TimeDiff { get; set; }

        public Image<Gray, double> Translation { get; set; } // 3-vector
        public Image<Gray, double> TranslationDiff { get; set; } // 3-vector

        public Image<Gray, double> Rotation { get; set; } // 3-vector - euler angles (xyz) in degrees
        public Image<Gray, double> RotationDiff { get; set; } // 3-vector - euler angles (xyz) in degrees

        public Image<Gray, double> Velocity { get; set; } // 3-vector
        public Image<Gray, double> AngularVelocity { get; set; } // 3-vector
    }

    public interface IVisualOdometer
    {
        OdometerFrame ComputeOdometry(Mat frame1, Mat frame2);

        // Draws results of last computations on given image (it should be same as frame2 above)
        void Visualize(Image<Bgr, byte> image);
    }
}
