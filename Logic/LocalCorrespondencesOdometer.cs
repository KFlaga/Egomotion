using Emgu.CV;
using Emgu.CV.Structure;
using System.Collections.Generic;

namespace Egomotion
{
    public class LocalCorrespondencesOdometer : IVisualOdometer
    {
        public OdometerFrame ComputeOdometry(Mat frame1, Mat frame2)
        {
            // TODO
            return default(OdometerFrame);
        }

        public void Visualize(Image<Bgr, byte> image)
        {
        }
    }
}
