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
    public class MacthingResult
    {
        public VectorOfPointF LeftPoints { get; set; }
        public VectorOfPointF RightPoints { get; set; }
        public List<double> Distances { get; set; }

        public MKeyPoint[] LeftKps { get; set; }
        public MKeyPoint[] RightKps { get; set; }
        public VectorOfDMatch Matches { get; set; }
    }

    public static class MatchImagePair
    {
        public static void FindFeatures(Mat image, Feature2D detector, out MKeyPoint[] kps, out Mat descriptors)
        {
            kps = detector.Detect(image);

            VectorOfKeyPoint vectorOfKp = new VectorOfKeyPoint(kps);

            var desc = new Emgu.CV.XFeatures2D.BriefDescriptorExtractor(32);
            descriptors = new Mat();
            desc.Compute(image, vectorOfKp, descriptors);
        }

        public static VectorOfDMatch FindMatches(Mat desc1, Mat desc2)
        {
            BFMatcher m = new BFMatcher(DistanceType.Hamming, true);
            VectorOfDMatch matches = new VectorOfDMatch();
            m.Match(desc1, desc2, matches);
            return matches;
        }

        public static void MacthesToPointLists(VectorOfDMatch matches, MKeyPoint[] kp1, MKeyPoint[] kp2,
            out VectorOfPointF leftPoints, out VectorOfPointF rightPoints, out List<double> distances)
        {
            leftPoints = new VectorOfPointF();
            rightPoints = new VectorOfPointF();
            distances = new List<double>();

            var sortedMatches = matches.ToArray().OrderBy((x) => x.Distance);
            foreach(var m in sortedMatches)
            {
                leftPoints.Push(new PointF[] { kp1[m.QueryIdx].Point });
                rightPoints.Push(new PointF[] { kp2[m.TrainIdx].Point });
                distances.Add(m.Distance);
            }
        }

        public static MacthingResult Match(Mat left, Mat right, Feature2D detector)
        {
            FindFeatures(left, detector, out MKeyPoint[] kps1, out Mat desc1);
            FindFeatures(right, detector, out MKeyPoint[] kps2, out Mat desc2);

            var matches = FindMatches(desc1, desc2);

            MacthesToPointLists(matches, kps1, kps2, out VectorOfPointF leftPoints, out VectorOfPointF rightPoints, out List<double> distances);

            return new MacthingResult()
            {
                LeftPoints = leftPoints,
                RightPoints = rightPoints,
                LeftKps = kps1,
                RightKps = kps2,
                Matches = matches,
                Distances = distances
            };
        }
    }
}
