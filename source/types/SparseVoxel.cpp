#include "localization/types/SparseVoxel.h"
#include <algorithm>
#include <cmath>
#include <execution>
#include <ranges>
#include <boost/unordered/unordered_flat_set.hpp>

namespace
{
bool outsideSearchRadius(const float lower_bound, const float squared_radius)
{
    // Keep a conservative margin for float rounding at voxel faces.
    constexpr float rounding_slack = 1.0f + 4.0f * std::numeric_limits<float>::epsilon();
    return lower_bound > squared_radius * rounding_slack;
}
}

bool SparseVoxel::insert(const pcl::PointCloud<pcl::PointXYZ>& cloud)
{
    if (cloud.empty())
    {
        return false;
    }

    boost::unordered_flat_set<VoxelKey> updated_voxels;
    for (const auto& point : cloud)
    {
        const Eigen::Vector3f eigen_point = point.getVector3fMap();
        const VoxelKey key = pointToVoxel(eigen_point);

        auto [it, inserted] = mOccupiedVoxels.try_emplace(key, mCells.size());
        if (inserted)
        {
            mCells.emplace_back();
        }

        if (mCells[it->second].add(eigen_point) != nullptr)
        {
            updated_voxels.insert(key);
            ++mPointCount;
        }
    }

    boost::unordered_flat_set<std::size_t> affected_cells;
    for (const auto& [i, j, k] : updated_voxels)
    {
        constexpr int voxel_radius = 1;
        for (int dx = -voxel_radius; dx <= voxel_radius; ++dx)
        {
            for (int dy = -voxel_radius; dy <= voxel_radius; ++dy)
            {
                for (int dz = -voxel_radius; dz <= voxel_radius; ++dz)
                {
                    if (const auto found = mOccupiedVoxels.find(VoxelKey{i + dx, j + dy, k + dz});
                        found != mOccupiedVoxels.end())
                    {
                        affected_cells.insert(found->second);
                    }
                }
            }
        }
    }

    const std::vector<std::size_t> cells(affected_cells.begin(), affected_cells.end());
    estimateCovariances(cells);
	return true;
}

void SparseVoxel::estimateCovariances(const std::vector<std::size_t>& cells)
{
	std::for_each(std::execution::par, cells.begin(), cells.end(),
		[this](const std::size_t cell_index)
		{
			VoxelCell& cell = mCells[cell_index];
			const VoxelKey key = pointToVoxel(cell.points.front().position);
			const NeighborCells candidates = neighborCells(key);
			for (PointWithCovariance& point : cell.points)
			{
				point.covariance = Eigen::Matrix3f::Identity();
				point.covariance_valid = false;

				const std::vector<Neighbor> neighbors = nearestNeighbors(point.position, MAX_COVARIANCE_NEIGHBORS, key, candidates);
				if (static_cast<int>(neighbors.size()) < MIN_COVARIANCE_NEIGHBORS)
				{
					continue;
				}

				Eigen::Vector3f mean = Eigen::Vector3f::Zero();
				for (const auto& [neighbor, _] : neighbors)
				{
					mean += neighbor->position;
				}
				mean /= static_cast<float>(neighbors.size());

				Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
				for (const auto& [neighbor, _] : neighbors)
				{
					const Eigen::Vector3f offset = neighbor->position - mean;
					covariance.noalias() += offset * offset.transpose();
				}
				covariance /= static_cast<float>(neighbors.size() - 1);
			    constexpr float diagonal_regularization = 1.0e-6f;
			    covariance.diagonal().array() += diagonal_regularization;

                const float inverse_norm = covariance.inverse().norm();
                point.covariance = covariance * inverse_norm;
				point.covariance_valid = true;
			}
		});
}

std::array<float, 27> SparseVoxel::voxelDistanceBounds(const Eigen::Vector3f& query, const VoxelKey& key)
{
    const Eigen::Vector3f origin = Eigen::Vector3f(static_cast<float>(key.i), static_cast<float>(key.j),
                                                  static_cast<float>(key.k)) * VOXEL_SIZE;
    const Eigen::Vector3f below = (query - origin).cwiseMax(0.0f);
    const Eigen::Vector3f above = (origin + Eigen::Vector3f::Constant(VOXEL_SIZE) - query).cwiseMax(0.0f);
    const std::array<float, 3> x_distances{below.x(), 0.0f, above.x()};
    const std::array<float, 3> y_distances{below.y(), 0.0f, above.y()};
    const std::array<float, 3> z_distances{below.z(), 0.0f, above.z()};
    std::array<float, 27> bounds{};
    std::size_t offset = 0;
    for (float x : x_distances)
    {
        for (float y : y_distances)
        {
            for (float z : z_distances)
            {
                bounds[offset++] = Eigen::Vector3f(x, y, z).squaredNorm();
            }
        }
    }
    return bounds;
}

SparseVoxel::NeighborCells SparseVoxel::neighborCells(const VoxelKey& key) const
{
    static constexpr auto offsets = []
    {
        std::array<std::array<int, 3>, 27> result{};
        std::size_t next = 0;
        for (int order = 0; order <= 3; ++order)
        {
            for (int dx = -1; dx <= 1; ++dx)
            {
                for (int dy = -1; dy <= 1; ++dy)
                {
                    for (int dz = -1; dz <= 1; ++dz)
                    {
                        if (dx * dx + dy * dy + dz * dz == order)
                        {
                            result[next++] = {dx, dy, dz};
                        }
                    }
                }
            }
        }
        return result;
    }();

    NeighborCells cells;
    for (const auto& [dx, dy, dz] : offsets)
    {
        if (const auto found = mOccupiedVoxels.find(VoxelKey{key.i + dx, key.j + dy, key.k + dz});
            found != mOccupiedVoxels.end())
        {
            const int offset = (dx + 1) * 9 + (dy + 1) * 3 + dz + 1;
            cells.push_back(NeighborCell{&mCells[found->second], offset});
        }
    }
    return cells;
}

std::vector<const PointWithCovariance*> SparseVoxel::points() const
{
	std::vector<const PointWithCovariance*> points;
	points.reserve(mPointCount);
    for (const VoxelCell& cell : mCells)
	{
		for (const PointWithCovariance& point : cell.points)
		{
			points.push_back(&point);
		}
	}
	return points;
}

SparseVoxel::Neighbor SparseVoxel::nearestNeighbor(const Eigen::Vector3f& query, const float max_distance) const
{
	Neighbor nearest;
	nearest.squared_distance = max_distance * max_distance;
	const VoxelKey key = pointToVoxel(query);
	const auto [i, j, k] = key;
	const auto bounds = voxelDistanceBounds(query, key);
	int best_offset = 27;
	const auto search_cell = [&](const VoxelCell& cell, int offset)
	{
		for (const PointWithCovariance& point : cell.points)
		{
			const float squared_distance = (query - point.position).squaredNorm();
			if (squared_distance < nearest.squared_distance ||
				(nearest.point != nullptr && squared_distance == nearest.squared_distance && offset < best_offset))
			{
				nearest = Neighbor{&point, squared_distance};
				best_offset = offset;
			}
		}
	};
	if (const auto found = mOccupiedVoxels.find(key); found != mOccupiedVoxels.end())
	{
		search_cell(mCells[found->second], 13);
	}

    constexpr int voxel_radius = 1;
	int offset = 0;
	for (int dx = -voxel_radius; dx <= voxel_radius; ++dx)
	{
		for (int dy = -voxel_radius; dy <= voxel_radius; ++dy)
		{
			for (int dz = -voxel_radius; dz <= voxel_radius; ++dz, ++offset)
			{
				if (offset == 13 || outsideSearchRadius(bounds[static_cast<std::size_t>(offset)], nearest.squared_distance))
				{
					continue;
				}
				const auto found = mOccupiedVoxels.find(VoxelKey{i + dx, j + dy, k + dz});
				if (found == mOccupiedVoxels.end())
				{
					continue;
				}

				search_cell(mCells[found->second], offset);
			}
		}
	}
	return nearest;
}

std::vector<SparseVoxel::Neighbor> SparseVoxel::nearestNeighbors(const Eigen::Vector3f& query, const int max_neighbors) const
{
    if (max_neighbors <= 0)
    {
        return {};
    }
    const VoxelKey key = pointToVoxel(query);
    return nearestNeighbors(query, max_neighbors, key, neighborCells(key));
}

std::vector<SparseVoxel::Neighbor> SparseVoxel::nearestNeighbors(const Eigen::Vector3f& query, const int max_neighbors,
                                                                 const VoxelKey& key, const NeighborCells& cells)
{
    std::vector<Neighbor> neighbors;
    neighbors.reserve(static_cast<std::size_t>(max_neighbors));

    const auto bounds = voxelDistanceBounds(query, key);

    // Max-heap:
    // neighbors.front() is always the farthest neighbor
    // among the current top-K nearest neighbors.
    const auto compare = [](const Neighbor& lhs, const Neighbor& rhs)
    {
        return lhs.squared_distance < rhs.squared_distance;
    };

    for (const NeighborCell& candidate : cells)
    {
        if (neighbors.size() == static_cast<std::size_t>(max_neighbors) &&
            outsideSearchRadius(bounds[static_cast<std::size_t>(candidate.offset)], neighbors.front().squared_distance))
        {
            continue;
        }
        for (const PointWithCovariance& point : candidate.cell->points)
        {
            const float squared_distance = (query - point.position).squaredNorm();
            if (neighbors.size() < static_cast<std::size_t>(max_neighbors))
            {
                neighbors.push_back(Neighbor{.point = &point, .squared_distance = squared_distance});
                std::ranges::push_heap(neighbors, compare);
                continue;
            }
            if (squared_distance >= neighbors.front().squared_distance)
            {
                continue;
            }
            std::ranges::pop_heap(neighbors, compare);
            neighbors.back() = Neighbor{.point = &point, .squared_distance = squared_distance};
            std::ranges::push_heap(neighbors, compare);
        }
    }

    return neighbors;
}