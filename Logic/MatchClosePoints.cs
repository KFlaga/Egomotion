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
    public static class MatchClosePoints
    {
        public static Func<Item, Item, double> GetDistanceFunc(DistanceType distanceType)
        {
            switch(distanceType)
            {
                case DistanceType.Hamming:
                    return (row1, row2) => CvInvoke.Norm(row1.desc, row2.desc, (Emgu.CV.CvEnum.NormType)6);
                case DistanceType.L2:
                    return (row1, row2) => CvInvoke.Norm(row1.desc, row2.desc, (Emgu.CV.CvEnum.NormType)4);
                case DistanceType.L1:
                    return (row1, row2) => CvInvoke.Norm(row1.desc, row2.desc, (Emgu.CV.CvEnum.NormType)2);
                default:
                    return (row1, row2) => CvInvoke.Norm(row1.desc, row2.desc, (Emgu.CV.CvEnum.NormType)4);
            }
        }

        public struct Item
        {
            public int index;
            public PointF pos;
            public Mat desc;
        }

        public class CompareItems : IComparer<Item>
        {
            public int Compare(Item x, Item y)
            {
                return x.pos.X < y.pos.X ? -1 :
                       x.pos.X > y.pos.X ? 1 :
                       0;
            }
        }

        public static double GetDistance(PointF p1, PointF p2)
        {
            double dx = p1.X - p2.X;
            double dy = p1.Y - p2.Y;
            return dx * dx + dy * dy;
        }

        public class WithMaxDistance : IComparer<Item>
        {
            double d;

            public WithMaxDistance(double dist)
            {
                d = dist;
            }

            public int Compare(Item x, Item y)
            {
                double dx = x.pos.X - y.pos.X;
                double dy = x.pos.Y - y.pos.Y;
                return dx * dx + dy * dy < d * d ? 0 :
                       dx > 0 ? 1 : -1;
            }
        }

        public static List<Item> SortByX(MKeyPoint[] kps, Mat desc)
        {
            List<Item> points = new List<Item>(kps.Length);
            for (int i = 0; i < kps.Length; ++i)
            {
                points.Add(new Item { index = i, pos = kps[i].Point, desc = desc.Row(i) });
            }
            points.Sort(new CompareItems());
            return points;
        }

        public static List<Item> SortByX(VectorOfPointF kps)
        {
            List<Item> points = new List<Item>(kps.Size);
            for (int i = 0; i < kps.Size; ++i)
            {
                points.Add(new Item { index = i, pos = kps[i] });
            }
            points.Sort(new CompareItems());
            return points;
        }

        public static int FindBestMatch(Item kp1, List<Item> kps2, Func<Item, Item, double> distance, double maxDistance = 20.0)
        {
            int firstClose = LowerBound(kps2, kp1, new WithMaxDistance(maxDistance));
            if (firstClose < kps2.Count)
            {
                var kp2 = kps2[firstClose];
                int bestMatch = -1;
                double bestCost = 1e8;
                while (Math.Abs(kp1.pos.X - kp2.pos.X) < maxDistance)
                {
                    if (GetDistance(kp1.pos, kp2.pos) < maxDistance * maxDistance)
                    {
                        double cost = distance(kp1, kp2);
                        if (cost < bestCost)
                        {
                            bestCost = cost;
                            bestMatch = firstClose;
                        }
                    }

                    ++firstClose;
                    if (firstClose >= kps2.Count)
                        break;
                    kp2 = kps2[firstClose];
                }
                return bestMatch;
            }
            return -1;
        }

        public static List<MDMatch> Match(MKeyPoint[] kpsQuery, MKeyPoint[] kpsTrain, Mat descQuery, Mat descTrain, DistanceType distanceType, double maxDistance = 20.0, bool crossCheck = true)
        {
            var matches = new List<MDMatch>(kpsTrain.Length / 2); 
            var distance = GetDistanceFunc(distanceType);
            var limitDistance = new WithMaxDistance(maxDistance);

            var kps1 = SortByX(kpsQuery, descQuery);
            var kps2 = SortByX(kpsTrain, descTrain);
            
            for (int i = 0; i < kps1.Count; ++i)
            {
                var kp1 = kps1[i];
                int bestMatch = FindBestMatch(kp1, kps2, distance, maxDistance);
                if (bestMatch >= 0)
                {
                    var kp2 = kps2[bestMatch];
                    if (crossCheck)
                    {
                        int best2To1 = FindBestMatch(kp2, kps1, distance, maxDistance);
                        if(best2To1 < 0 || best2To1 >= kps1.Count || kps1[best2To1].index != kp1.index)
                        {
                            continue;
                        }
                    }

                    matches.Add(new MDMatch()
                    {
                        Distance = (float)distance(kp1, kp2),
                        ImgIdx = 0,
                        TrainIdx = kp2.index,
                        QueryIdx = kp1.index
                    });
                }
            }


            return matches;
        }

        public static int LowerBound<T>(this IList<T> sortedCollection, T key, IComparer<T> comparer)
        {
            int begin = 0;
            int end = sortedCollection.Count;
            while (end > begin)
            {
                int index = (begin + end) / 2;
                T el = sortedCollection[index];
                if (comparer.Compare(el, key) >= 0)
                    end = index;
                else
                    begin = index + 1;
            }
            return end;
        }

    }
}
