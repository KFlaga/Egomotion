using Emgu.CV;
using Emgu.CV.Features2D;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using MathNet.Numerics.Optimization;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using Matrix = Emgu.CV.Image<Egomotion.Arthmetic, double>;

namespace Egomotion
{
    public class ScaleBy3dPointsMatch
    {
        public Matrix K { get; set; }
        public double RotationTreshold { get; set; } = 0.02;
        public int MinimumCorrespondencesNeeded { get; set; } = 8;

        int lastGoodLeftImage;
        int lastGoodRightImage;
        MatchingResult lastGoodMatch;
        Matrix R12;
        Matrix t12;
        Matrix c12;
        Matrix last3dPoints;

        bool isContinuous;

        public OdometerFrame NextFrame(int left, int right, Func<int, int, MatchingResult> matcher)
        {
            MatchingResult match23 = matcher(left, right);
            if (match23.Matches.Size < MinimumCorrespondencesNeeded)
            {
                // Track of points is lost, at least temporarly. Let's put handling lost-track case ou of scope for now.
                lastGoodMatch = null;
                isContinuous = false;
                return null;
            }

            OdometerFrame frame = new OdometerFrame()
            {
                MatK = K,
                Match = match23,
                Rotation = new Image<Arthmetic, double>(1, 3),
                RotationMatrix = RotationConverter.EulerXYZToMatrix(new Image<Arthmetic, double>(1, 3)),
                Translation = new Image<Arthmetic, double>(1, 3)
            };

            // 1) Determine if transformation between next frames has high enough baseline to be accurate

            // 1a) For now lets determine it by finding if lone rotation is good enough
            var H = FindTransformation.EstimateHomography(match23.LeftPointsList, match23.RightPointsList, K);
            if (FindTransformation.IsPureRotation(H, RotationTreshold))
            {
                // 1c) If not then transformation is described only by rotation
                // 1b) Find rotation and rotate all points in current set
                isContinuous = false;
                frame.Rotation = RotationConverter.MatrixToEulerXYZ(H);
                frame.RotationMatrix = RotationConverter.EulerXYZToMatrix(frame.Rotation);

                if(last3dPoints != null && R12 != null)
                {
                    last3dPoints = Errors.PutRTo4x4(frame.RotationMatrix).Multiply(last3dPoints);
                    R12 = frame.RotationMatrix.Multiply(R12);
                }
                else
                {
                    R12 = frame.RotationMatrix;
                }

                // 1c) Skip frame and wait for next one (but save matches)
                return frame;
            }

            // 2) We have legit frames
            if (!FindTransformation.FindTwoViewsMatrices(match23.LeftPointsList, match23.RightPointsList, K,
                out var F23, out var E23, out var R23, out var t23, out var X23))
            {
                // 3a) Or not
                isContinuous = false;
                return null;
            }
            // Normalize to |t| = 1
            t23 = t23.Mul(1.0 / t23.Norm);

            frame.Rotation = RotationConverter.MatrixToEulerXYZ(R23);
            frame.RotationMatrix = R23;
            frame.Translation = t23;

            // 3) Find same points between old frame and current one
            if (lastGoodMatch == null)
            {
                last3dPoints = X23;
                isContinuous = true;
            }
            else
            {
                #region NonContinousCase
                //if (!isContinuous) // This doesn't work well. Lets put it out of scope and just reset scale 
                // {
                // Find correspondences between last right and new left
                //var match12 = lastGoodMatch;
                //var match34 = match23;
                //var match23_ = matcher(lastGoodRightImage, left); // TODO: make use of already found feature points

                //var correspondences23to34 = Correspondences.FindCorrespondences12to23(match23_, match34);

                //// Now extend each correspondence to 4 points - find if point on 2 is matched to some point on 1
                //var correspondences13to34 = new List<Correspondences.MatchPair>();
                //foreach(var c in correspondences23to34)
                //{
                //    var m23 = c.Match12;
                //    for (int i = 0; i < match12.Matches.Size; ++i)
                //    {
                //        if(match12.Matches[i].TrainIdx == m23.QueryIdx)
                //        {
                //            correspondences13to34.Add(new Correspondences.MatchPair()
                //            {
                //                Kp1 = match12.LeftKps[match12.Matches[i].QueryIdx],
                //                Kp2 = c.Kp2,
                //                Kp3 = c.Kp3
                //            });
                //        }
                //    }
                //}

                //if (correspondences13to34.Count >= MinimumCorrespondencesNeeded)
                //{
                //    var t13 = R12.Multiply(c12).Mul(-1);

                //    FindBestScale(R12, t13, R23, t23, K, correspondences13to34, MinimumCorrespondencesNeeded, out double scale, out double confidence, out List<int> inliers);

                //    t23 = t23.Mul(scale);
                //    frame.Translation = t23;

                //    FindTransformation.TriangulateChieral(match23.LeftPointsList, match23.RightPointsList, K, R23, t23, out last3dPoints);

                //    isContinuous = true;
                //}
                //else
                //{
                //    isContinuous = false;
                //}
                //  }
                #endregion
                if (isContinuous)
                {
                    var correspondences = Correspondences.FindCorrespondences12to23(lastGoodMatch, match23);
                    if (correspondences.Count >= MinimumCorrespondencesNeeded)
                    {
                        FindBestScale(R12, t12, R23, t23, K, correspondences, MinimumCorrespondencesNeeded, out double scale, out double confidence, out List<int> inliers);

                        t23 = t23.Mul(scale);
                        frame.Translation = t23;

                        FindTransformation.TriangulateChieral(match23.LeftPointsList, match23.RightPointsList, K, R23, t23, out last3dPoints);
                    }
                    else
                    {
                        isContinuous = false;
                    }
                }

            }

            lastGoodMatch = match23;
            lastGoodLeftImage = left;
            lastGoodRightImage = right;
            R12 = R23;
            t12 = t23;
            c12 = R23.T().Multiply(t23).Mul(-1);

            return frame;
        }

        public static void FindBestScale(Matrix R12, Matrix t12, Matrix R23, Matrix t23, Matrix K, List<Correspondences.MatchPair> correspondences, int minSampleSize,
            out double scale, out double confidence, out List<int> inliers)
        {
            var pts1 = correspondences.Select((x) => x.Kp1.Point).ToList();
            var pts2 = correspondences.Select((x) => x.Kp2.Point).ToList();
            var pts3 = correspondences.Select((x) => x.Kp3.Point).ToList();

            FindTransformation.TriangulateChieral(pts1, pts2, K, R12, t12, out var est3d_12);
            FindTransformation.TriangulateChieral(pts2, pts3, K, R23, t23, out var est3d_23);

            // Find best scale, so that both sets will be closest
            RansacScaleEstimation ransacModel = new RansacScaleEstimation(est3d_12, est3d_23, R12, ComputeMatrix.Center(t12, R12));

            int sampleSize = Math.Max(minSampleSize, (int)(0.05 * correspondences.Count));
            int minGoodPoints = (int)(0.4 * correspondences.Count);
            int maxIterations = 100;
            double meanRefPointSize = GetMeanSize(est3d_12);
            double threshold = meanRefPointSize * meanRefPointSize * 0.08;
            var result = RANSAC.ProcessMostInliers(ransacModel, maxIterations, sampleSize, minGoodPoints, threshold);
            scale = (double)result.BestModel;
            inliers = result.Inliers;

            var backprojected23to12 = TransfromBack3dPoints(R12, t12, est3d_23, scale);

            Image<Arthmetic, double> inliersOnly12 = new Image<Arthmetic, double>(result.Inliers.Count, 4);
            Image<Arthmetic, double> inliersOnly23 = new Image<Arthmetic, double>(result.Inliers.Count, 4);
            for (int i = 0; i < result.Inliers.Count; ++i)
            {
                int k = result.Inliers[i];
                for (int j = 0; j < 4; ++j)
                {
                    inliersOnly12[j, i] = est3d_12[j, k];
                    inliersOnly23[j, i] = backprojected23to12[j, k];
                }
            }

            // Find errors
            var distances = inliersOnly12.Sub(inliersOnly23);
            double meanDistance = distances.Norm / distances.Cols;
            double meanSize = GetMeanSize(inliersOnly12, inliersOnly23);
            double relativeMeanDistance = meanDistance / meanSize;
            double error = result.BestError;
            double relativeError = error / (meanSize * meanSize);
            Errors.TraingulationError(inliersOnly12, inliersOnly23, out double mean, out double median, out List<double> errs);

            confidence = (double)inliers.Count / (double)correspondences.Count;
        }

        public static double GetMeanSize(Matrix pts12, Matrix pts23)
        {
            return GetMeanSize(pts12) + GetMeanSize(pts23);
        }

        public static double GetMeanSize(Matrix pts)
        {
            double s = 0.0;
            for (int i = 0; i < pts.Cols; ++i)
            {
                s += Math.Abs(pts[0, i]);
                s += Math.Abs(pts[1, i]);
                s += Math.Abs(pts[2, i]);
            }
            return s / (pts.Cols * 2);
        }

        public static Matrix TransfromBack3dPoints(Matrix R, Matrix t, Matrix pts23, double scale)
        {
            var backprojected = Errors.PutRTo4x4(R.T()).Multiply(pts23);
            var C = ComputeMatrix.Center(t, R);
            for (int i = 0; i < backprojected.Cols; ++i)
            {
                backprojected[0, i] = backprojected[0, i] * scale + C[0, 0];
                backprojected[1, i] = backprojected[1, i] * scale + C[1, 0];
                backprojected[2, i] = backprojected[2, i] * scale + C[2, 0];
            }
            return backprojected;
        }
    }

    public static class Correspondences
    {
        public static List<MatchPair> FindCorrespondences12to23(MatchingResult match12, MatchingResult match23)
        {
            Dictionary<int, int> kpIndexToMatchIndexOld = new Dictionary<int, int>(match12.Matches.Size);
            for (int i = 0; i < match12.Matches.Size; ++i)
            {
                kpIndexToMatchIndexOld[match12.Matches[i].TrainIdx] = i;
            }

            List<MatchPair> correspondences = new List<MatchPair>(match12.Matches.Size / 2);
            for (int i = 0; i < match23.Matches.Size; ++i)
            {
                int kpIndex = match23.Matches[i].QueryIdx;
                if (kpIndexToMatchIndexOld.TryGetValue(kpIndex, out int oldIndex))
                {
                    var oldMatch = match12.Matches[oldIndex];
                    var newMatch = match23.Matches[i];
                    correspondences.Add(new MatchPair()
                    {
                        Match12 = oldMatch,
                        Match23 = newMatch,
                        Kp1 = match12.LeftKps[oldMatch.QueryIdx],
                        Kp2 = match23.LeftKps[newMatch.QueryIdx],
                        Kp3 = match23.RightKps[newMatch.TrainIdx]
                    });
                }
            }
            return correspondences;
        }

        public static List<MatchPair> FindCorrespondences12to23to31(MatchingResult match12, MatchingResult match23, MatchingResult match31)
        {
            var c12to23 = FindCorrespondences12to23(match12, match23);
            var crossCorrespondences = new List<MatchPair>(c12to23.Count);

            Dictionary<int, int> kpIndex3to1 = new Dictionary<int, int>(match31.Matches.Size);
            for (int i = 0; i < match31.Matches.Size; ++i)
            {
                kpIndex3to1[match31.Matches[i].QueryIdx] = match31.Matches[i].TrainIdx;
            }

            foreach (var c in c12to23)
            {
                // Accept correspondences only if we have corresponding match from 3 to 1
                int expectedKp1 = c.Match12.QueryIdx;
                int expectedKp3 = c.Match23.TrainIdx;
                if (kpIndex3to1.TryGetValue(expectedKp3, out int kp1) && kp1 == expectedKp1)
                {
                    crossCorrespondences.Add(c);
                }
            }
            return crossCorrespondences;
        }

        public class MatchPair
        {
            public MDMatch Match12;
            public MDMatch Match23;

            public MKeyPoint Kp1;
            public MKeyPoint Kp2;
            public MKeyPoint Kp3;
        }

        public struct SortItem
        {
            public PointF pos;
            public int index;
        }

        public class XYSorter : IComparer<SortItem>
        {
            public int Compare(SortItem x, SortItem y)
            {
                return x.pos.X < y.pos.X ? -1 :
                       x.pos.X > y.pos.X ? 1 :
                       x.pos.Y < y.pos.Y ? -1 :
                       x.pos.Y > y.pos.Y ? 1 :
                       0;
            }
        }

        public static List<SortItem> SortByXY(MKeyPoint[] kps)
        {
            List<SortItem> points = new List<SortItem>(kps.Length);
            for (int i = 0; i < kps.Length; ++i)
            {
                points.Add(new SortItem { index = i, pos = kps[i].Point });
            }
            points.Sort(new XYSorter());
            return points;
        }
    }
    
    public class RansacScaleEstimation : IRansacModel<int>
    {
        public Image<Arthmetic, double> Pts12 { get; set; }
        public Image<Arthmetic, double> Pts23 { get; set; }
        public List<int> RansacPoints { get; set; }

        public IList<int> AllPoints => RansacPoints;

        public RansacScaleEstimation(Image<Arthmetic, double> pts12, Image<Arthmetic, double> pts23, Image<Arthmetic, double> R, Image<Arthmetic, double> C)
        {
            Pts23 = Errors.PutRTo4x4(R.T()).Multiply(pts23);
            Pts12 = pts12.Clone();
            RansacPoints = new List<int>(Pts12.Cols);
            for (int i = 0; i < Pts12.Cols; ++i)
            {
                Pts12[0, i] = Pts12[0, i] / Pts12[3, i] - C[0, 0];
                Pts12[1, i] = Pts12[1, i] / Pts12[3, i] - C[1, 0];
                Pts12[2, i] = Pts12[2, i] / Pts12[3, i] - C[2, 0];
                RansacPoints.Add(i);
            }
        }

        public object FindModel(IEnumerable<int> points)
        {
            // Find scale s so that Pts12 - s * Pts23 == 0. Let X = Pts12, Y = Pts23
            // We can minimize error: (x1 - sy1)^2 + (x2 - sy)^2 + ...
            // Derivative is: -2x1y1 + 2sy1^2 + ... = 0
            // So: s = sum(xiyi) / sum(yi^2)

            double xiyi = 0.0;
            double yi2 = 0.0;
            foreach(var p in points)
            {
                for(int i = 0; i < 3; ++i)
                {
                    xiyi += Pts12[i, p] * Pts23[i, p];
                    yi2 += Pts23[i, p] * Pts23[i, p];
                }
            }
            double scale = xiyi / yi2;
            return scale;
        }

        public double PointError(int p, object model)
        {
            double scale = (double)model;
            double error = 0.0;
            for(int i = 0; i < 3; ++i)
            {
                double d = Pts12[i, p] - scale * Pts23[i, p];
                error += d * d;
            }
            return error;
        }

        public double TotalError(double scale)
        {
            double error = 0.0;
            for(int  p = 0; p < Pts12.Cols; ++p)
            {
                for(int i = 0; i < 3; ++i)
                {
                    double d = Pts12[i, p] - scale * Pts23[i, p];
                    error += d * d;
                }
            }
            return error;
        }
    }
}
