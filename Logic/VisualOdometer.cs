using Emgu.CV;
using Emgu.CV.Structure;
using System;
using System.Collections.Generic;

namespace Egomotion
{
    public class OdometerFrame
    {
        public TimeSpan TimeDiff { get; set; }

        public Image<Arthmetic, double> Translation { get; set; } // column 3-vector
        public Image<Arthmetic, double> TranslationDiff { get; set; } // column 3-vector
        public Image<Arthmetic, double> Center { get; set; } // column 3-vector

        public Image<Arthmetic, double> Rotation { get; set; } // column 3-vector - euler angles (xyz) in degrees
        public Image<Arthmetic, double> RotationDiff { get; set; } // column 3-vector - euler angles (xyz) in degrees
        public Image<Arthmetic, double> RotationMatrix { get; set; } // 


        public Image<Arthmetic, double> Velocity { get; set; } // column 3-vector
        public Image<Arthmetic, double> AngularVelocity { get; set; } // column 3-vector

        public Image<Arthmetic, double> MatK { get; set; } // K

        public MacthingResult Match { get; set; }
    }

    public interface IVisualOdometer
    {
        OdometerFrame ComputeOdometry(Mat frame1, Mat frame2);

        // Draws results of last computations on given image (it should be same as frame2 above)
        void Visualize(Image<Bgr, byte> image);
    }
}
