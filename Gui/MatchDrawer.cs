using Emgu.CV;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Egomotion
{
    public class MatchDrawer
    {
        public static void DrawFeatures(Mat left, Mat right, MacthingResult match, double takeBest, ImageViewer macthedView)
        {
            Mat matchesImage = new Mat();
            VectorOfVectorOfDMatch matches2 = new VectorOfVectorOfDMatch();
            VectorOfKeyPoint vectorOfKp2 = new VectorOfKeyPoint(match.LeftKps);
            VectorOfKeyPoint vectorOfKp1 = new VectorOfKeyPoint(match.RightKps);
            matches2.Push(new VectorOfDMatch(match.Matches.ToArray().OrderBy((x) => x.Distance).Take((int)(match.Matches.Size * takeBest)).ToArray()));
            // Features2DToolbox.DrawMatches(left, vectorOfKp1, right, vectorOfKp2, matches2, matchesImage, new Bgr(Color.Red).MCvScalar, new Bgr(Color.Blue).MCvScalar);
            Features2DToolbox.DrawMatches(right, vectorOfKp1, left, vectorOfKp2, matches2, matchesImage, new Bgr(Color.Red).MCvScalar, new Bgr(Color.Blue).MCvScalar);

            macthedView.Source = ImageLoader.ImageSourceForBitmap(matchesImage.Bitmap);
        }

        public static void DrawCricles(ImageViewer view, Mat image, MKeyPoint[] points)
        {
            var processedImage = image.Clone();
            foreach (var kp in points)
            {
                DrawCricle(processedImage, new Bgr(Color.Wheat), new System.Drawing.Point((int)kp.Point.X, (int)kp.Point.Y), new System.Drawing.Size(10, 10));
            }
            view.Source = ImageLoader.ImageSourceForBitmap(processedImage.Bitmap);
        }

        public static void DrawCricle(Mat image, Bgr color, System.Drawing.Point center, System.Drawing.Size size)
        {
            RotatedRect rect = new RotatedRect(center, size, 0);
            CvInvoke.Ellipse(image, rect, color.MCvScalar, 1);
        }
    }
}
