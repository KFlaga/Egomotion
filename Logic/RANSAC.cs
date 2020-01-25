using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Egomotion
{
    public interface IRansacModel<PointT>
    {
        IList<PointT> AllPoints { get; }
        object FindModel(IEnumerable<PointT> points);
        double PointError(PointT point, object model);
    }

    public class RANSAC
    {
        public class Results<PointT>
        {
            public object BestModel;
            public double BestError;
            public List<PointT> Inliers;
        }

        public static Results<PointT> Process<PointT>(IRansacModel<PointT> model, int maxIterations, int sampleSize, int minGoodPoints, double threshold, object initialModel)
        {
            Random random = new Random((int)DateTime.Now.Ticks);
            object bestModel = initialModel;
            double bestError = 10e8;
            
            for (int i = 0; i < maxIterations; ++i)
            {
                NextSample(model, sampleSize, threshold, random, out object sampleModel, out List<PointT> inliers);

                if (bestModel == null || inliers.Count >= minGoodPoints)
                {
                    sampleModel = model.FindModel(inliers);
                    double error = 0.0;
                    foreach (var point in model.AllPoints)
                    {
                        error += model.PointError(point, sampleModel);
                    }

                    if (error < bestError)
                    {
                        bestError = error;
                        bestModel = sampleModel;
                    }
                }
            }

            var bestInliers = model.AllPoints
                .Where((point) => model.PointError(point, bestModel) < threshold)
                .ToList();

            return new Results<PointT>()
            {
                BestModel = bestModel,
                BestError = bestError,
                Inliers = bestInliers
            };
        }

        private static void NextSample<PointT>(IRansacModel<PointT> model, int sampleSize, double threshold, Random random, out object sampleModel, out List<PointT> inliers)
        {
            IEnumerable<PointT> sample = PickRandomSample(model.AllPoints, sampleSize, random);
            sampleModel = model.FindModel(sample);
            inliers = new List<PointT>(model.AllPoints.Count);
            foreach (var point in model.AllPoints)
            {
                if (model.PointError(point, sampleModel) < threshold)
                {
                    inliers.Add(point);
                }
            }
        }

        public static Results<PointT> ProcessMostInliers<PointT>(IRansacModel<PointT> model, int maxIterations, int sampleSize, int minGoodPoints, double threshold, double initialModel)
        {
            Random random = new Random((int)DateTime.Now.Ticks);
            object bestModel = initialModel;
            List<PointT> bestInliers = new List<PointT>();

            for (int i = 0; i < maxIterations; ++i)
            {
                NextSample(model, sampleSize, threshold, random, out object sampleModel, out List<PointT> inliers);

                if (inliers.Count > bestInliers.Count)
                {
                    bestInliers = inliers;
                    bestModel = sampleModel;
                }
            }

            double bestError = 0.0;
            foreach (var point in model.AllPoints)
            {
                bestError += model.PointError(point, bestModel);
            }

            return new Results<PointT>()
            {
                BestModel = bestModel,
                BestError = bestError,
                Inliers = bestInliers
            };
        }

        public static IEnumerable<PointT> PickRandomSample<PointT>(IList<PointT> points, int sampleSize, Random random)
        {
            var items = new SortedSet<PointT>();
            while (sampleSize > 0)
                if (items.Add(points[random.Next(points.Count)]))
                    sampleSize--;
            return items;
        }
    }
}
