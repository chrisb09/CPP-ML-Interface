#pragma once

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iterator>
#include <memory>
#include <numeric>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ml_coupling_data_type.hpp"
#include "ml_coupling_memory_layout.hpp"

typedef enum
{
	MLCouplingOwnershipExternal = 0,
	MLCouplingOwnershipOwned = 1
} MLCouplingOwnership;

template <typename T>
class MLCouplingTensor
{
public:
	using value_type = T;

	class const_element_iterator
	{
	public:
		using iterator_category = std::forward_iterator_tag;
		using value_type = T;
		using difference_type = std::ptrdiff_t;
		using pointer = void;
		using reference = T;

		const_element_iterator(const MLCouplingTensor<T>* tensor, size_t linear_index)
			: tensor_(tensor), linear_index_(linear_index) {}

		T operator*() const
		{
			return tensor_->at_linear(linear_index_);
		}

		const_element_iterator& operator++()
		{
			++linear_index_;
			return *this;
		}

		const_element_iterator operator++(int)
		{
			const_element_iterator tmp(*this);
			++(*this);
			return tmp;
		}

		bool operator==(const const_element_iterator& other) const
		{
			return tensor_ == other.tensor_ && linear_index_ == other.linear_index_;
		}

		bool operator!=(const const_element_iterator& other) const
		{
			return !(*this == other);
		}

	private:
		const MLCouplingTensor<T>* tensor_;
		size_t linear_index_;
	};

	MLCouplingTensor() = default;

	MLCouplingTensor(void* root,
					 std::vector<int> dimensions,
					 MLCouplingMemoryLayout layout,
					 MLCouplingOwnership ownership = MLCouplingOwnershipExternal,
					 std::function<void(void*)> deleter = {})
		: root_(root),
		  dimensions_(std::move(dimensions)),
		  layout_(layout),
		  ownership_(ownership),
		  deleter_(std::move(deleter))
	{
		if (ownership_ == MLCouplingOwnershipOwned && root_ != nullptr)
		{
			owner_ = make_owner(root_, layout_, deleter_);
		}
		validate_shape();
	}

	static MLCouplingTensor<T> wrap_flat(T* data,
										 std::vector<int> dimensions,
										 MLCouplingMemoryLayout layout = MLCouplingMemLayoutContiguous,
										 MLCouplingOwnership ownership = MLCouplingOwnershipExternal)
	{
		if (!is_contiguous_layout(layout))
		{
			throw std::invalid_argument("wrap_flat(): layout must be contiguous");
		}
		return MLCouplingTensor<T>(static_cast<void*>(data), std::move(dimensions), layout, ownership,
								   ownership == MLCouplingOwnershipOwned
									   ? [](void* ptr) { delete[] static_cast<T*>(ptr); }
									   : std::function<void(void*)>{});
	}

	static MLCouplingTensor<T> wrap_nested(void* root,
										   std::vector<int> dimensions,
										   MLCouplingMemoryLayout layout = MLCouplingMemLayoutNested,
										   MLCouplingOwnership ownership = MLCouplingOwnershipExternal,
										   std::function<void(void*)> deleter = {})
	{
		if (!is_nested_layout(layout))
		{
			throw std::invalid_argument("wrap_nested(): layout must be nested");
		}
		return MLCouplingTensor<T>(root, std::move(dimensions), layout, ownership, std::move(deleter));
	}

	static MLCouplingTensor<T> from_flat_copy(const std::vector<T>& values,
											  const std::vector<int>& dimensions,
											  MLCouplingMemoryLayout layout = MLCouplingMemLayoutContiguous)
	{
		if (!is_contiguous_layout(layout))
		{
			throw std::invalid_argument("from_flat_copy(): layout must be contiguous");
		}

		const size_t expected = numel_from_shape(dimensions);
		if (expected != values.size())
		{
			throw std::invalid_argument("from_flat_copy(): values.size() does not match dimensions product");
		}

		T* buffer = new T[expected];
		std::copy(values.begin(), values.end(), buffer);
		return wrap_flat(buffer, dimensions, layout, MLCouplingOwnershipOwned);
	}

	~MLCouplingTensor()
	{
		release_owned();
	}

	MLCouplingTensor(const MLCouplingTensor& other)
		: root_(other.root_),
		  dimensions_(other.dimensions_),
		  layout_(other.layout_),
		  ownership_(other.ownership_),
		  deleter_(other.deleter_),
		  owner_(other.owner_) {}

	MLCouplingTensor& operator=(const MLCouplingTensor& other)
	{
		if (this == &other)
		{
			return *this;
		}
		release_owned();
		root_ = other.root_;
		dimensions_ = other.dimensions_;
		layout_ = other.layout_;
		ownership_ = other.ownership_;
		deleter_ = other.deleter_;
		owner_ = other.owner_;
		return *this;
	}

	MLCouplingTensor(MLCouplingTensor&& other) noexcept
		: root_(other.root_),
		  dimensions_(std::move(other.dimensions_)),
		  layout_(other.layout_),
		  ownership_(other.ownership_),
		  deleter_(std::move(other.deleter_)),
		  owner_(std::move(other.owner_))
	{
		other.root_ = nullptr;
		other.layout_ = MLCouplingMemLayoutInvalid;
		other.ownership_ = MLCouplingOwnershipExternal;
		other.deleter_ = {};
	}

	MLCouplingTensor& operator=(MLCouplingTensor&& other) noexcept
	{
		if (this == &other)
		{
			return *this;
		}

		release_owned();

		root_ = other.root_;
		dimensions_ = std::move(other.dimensions_);
		layout_ = other.layout_;
		ownership_ = other.ownership_;
		deleter_ = std::move(other.deleter_);
		owner_ = std::move(other.owner_);

		other.root_ = nullptr;
		other.layout_ = MLCouplingMemLayoutInvalid;
		other.ownership_ = MLCouplingOwnershipExternal;
		other.deleter_ = {};

		return *this;
	}
	const std::vector<int>& dimensions() const { return dimensions_; }
	// AIxeleratorService needs int64 dimensions, so we provide a helper for that. We can add similar helpers for other types if needed.
	const std::vector<int64_t> dimensions_as_int64() const
	{
		std::vector<int64_t> dims;
		dims.reserve(dimensions_.size());
		for (int dim : dimensions_)
		{
			dims.push_back(static_cast<int64_t>(dim));
		}
		return dims;
	}
	MLCouplingMemoryLayout layout() const { return layout_; }
	MLCouplingOwnership ownership() const { return ownership_; }
	void* root() const { return root_; }
	bool empty() const { return numel() == 0; }

	size_t rank() const
	{
		return dimensions_.size();
	}

	size_t numel() const
	{
		return numel_from_shape(dimensions_);
	}

    size_t element_size() const
    {
        return sizeof(T);
    }

	bool is_nested() const
	{
		return is_nested_layout(layout_);
	}

	bool is_contiguous() const
	{
		return is_contiguous_layout(layout_);
	}

	MLCouplingDataType data_type() const
	{
		return to_ml_coupling_data_type<T>();
	}

	T at(const std::vector<int>& index) const
	{
		validate_index(index);

		if (is_contiguous())
		{
			const size_t offset = linear_offset(index, dimensions_, layout_);
			return static_cast<T*>(root_)[offset];
		}

		if (dimensions_.size() == 1)
		{
			return static_cast<T*>(root_)[index[0]];
		}

		const void* current = root_;
		for (size_t depth = 0; depth + 1 < dimensions_.size(); ++depth)
		{
			const void* const* level = static_cast<const void* const*>(current);
			current = level[index[depth]];
		}
		const T* leaf = static_cast<const T*>(current);
		return leaf[index.back()];
	}

	T at_linear(size_t linear_index) const
	{
		if (linear_index >= numel())
		{
			throw std::out_of_range("at_linear(): linear index out of range");
		}
		return at(unravel_index(linear_index, dimensions_, layout_));
	}

	void set(const std::vector<int>& index, const T& value)
	{
		validate_index(index);

		if (is_contiguous())
		{
			const size_t offset = linear_offset(index, dimensions_, layout_);
			static_cast<T*>(root_)[offset] = value;
			return;
		}

		if (dimensions_.size() == 1)
		{
			static_cast<T*>(root_)[index[0]] = value;
			return;
		}

		void* current = root_;
		for (size_t depth = 0; depth + 1 < dimensions_.size(); ++depth)
		{
			void** level = static_cast<void**>(current);
			current = level[index[depth]];
		}
		T* leaf = static_cast<T*>(current);
		leaf[index.back()] = value;
	}

	void set_linear(size_t linear_index, const T& value)
	{
		if (linear_index >= numel())
		{
			throw std::out_of_range("set_linear(): linear index out of range");
		}
		set(unravel_index(linear_index, dimensions_, layout_), value);
	}

	std::vector<T> as_flat_vector(MLCouplingMemoryLayout target_layout = MLCouplingMemLayoutContiguous) const
	{
		if (!is_contiguous_layout(target_layout))
		{
			throw std::invalid_argument("as_flat_vector(): target_layout must be contiguous");
		}

		std::vector<T> flat(numel());
		for (size_t linear = 0; linear < flat.size(); ++linear)
		{
			const std::vector<int> idx = unravel_index(linear, dimensions_, target_layout);
			flat[linear] = at(idx);
		}
		return flat;
	}

	MLCouplingTensor<T> flatten(MLCouplingMemoryLayout target_layout = MLCouplingMemLayoutContiguous) const
	{
		std::vector<T> flat_values = as_flat_vector(target_layout);
		return from_flat_copy(flat_values, dimensions_, target_layout);
	}

	MLCouplingTensor<T> deep_copy() const
	{
		MLCouplingTensor<T> copy;
		copy.copy_from(*this);
		return copy;
	}

	MLCouplingTensor<T> unflatten(MLCouplingMemoryLayout target_layout = MLCouplingMemLayoutNested) const
	{
		if (!is_nested_layout(target_layout))
		{
			throw std::invalid_argument("unflatten(): target_layout must be nested");
		}

		const MLCouplingTensor<T> flat = flatten(to_contiguous_layout(target_layout));

		if (flat.rank() == 1)
		{
			const size_t n = flat.numel();
			T* leaf = new T[n];
			std::copy(static_cast<T*>(flat.root_), static_cast<T*>(flat.root_) + n, leaf);
			return wrap_nested(static_cast<void*>(leaf), flat.dimensions_, target_layout,
							   MLCouplingOwnershipOwned,
							   [](void* ptr) { delete[] static_cast<T*>(ptr); });
		}

		std::function<void*(size_t, size_t)> build_level;
		build_level = [&](size_t depth, size_t base_linear) -> void* {
			const int level_size = flat.dimensions_[depth];

			if (depth + 1 == flat.dimensions_.size() - 1)
			{
				void** table = new void*[level_size];
				const int leaf_size = flat.dimensions_.back();

				for (int i = 0; i < level_size; ++i)
				{
					T* leaf = new T[leaf_size];
					for (int j = 0; j < leaf_size; ++j)
					{
						std::vector<int> idx(flat.dimensions_.size(), 0);
						idx[depth] = i;
						idx.back() = j;

						const std::vector<int> prefix = unravel_index(base_linear, flat.dimensions_, flat.layout_);
						for (size_t p = 0; p < depth; ++p)
						{
							idx[p] = prefix[p];
						}

						const size_t src_off = linear_offset(idx, flat.dimensions_, flat.layout_);
						leaf[j] = static_cast<T*>(flat.root_)[src_off];
					}
					table[i] = static_cast<void*>(leaf);
				}
				return static_cast<void*>(table);
			}

			void** table = new void*[level_size];
			const size_t stride = stride_for_depth(flat.dimensions_, flat.layout_, depth);
			for (int i = 0; i < level_size; ++i)
			{
				table[i] = build_level(depth + 1, base_linear + static_cast<size_t>(i) * stride);
			}
			return static_cast<void*>(table);
		};

		void* root = build_level(0, 0);
		return wrap_nested(root,
						   flat.dimensions_,
						   target_layout,
						   MLCouplingOwnershipOwned,
						   [dims = flat.dimensions_](void* ptr) {
							   free_nested_tree(ptr, dims, 0);
						   });
	}

	const_element_iterator begin_elements() const { return const_element_iterator(this, 0); }
	const_element_iterator end_elements() const { return const_element_iterator(this, numel()); }

	std::string to_string(const std::string& class_name = "MLCouplingTensor") const
	{
		std::ostringstream out;
		out << class_name << "{"
			<< "dtype=" << ::to_string(data_type())
			<< ", layout=" << ::to_string(layout_)
			<< ", ownership=" << (ownership_ == MLCouplingOwnershipOwned ? "Owned" : "External")
			<< ", shape=[";
		for (size_t i = 0; i < dimensions_.size(); ++i)
		{
			if (i > 0)
			{
				out << ", ";
			}
			out << dimensions_[i];
		}
		out << "]";

		if (numel() <= 16)
		{
			out << ", values=[";
			bool first = true;
			for (auto it = begin_elements(); it != end_elements(); ++it)
			{
				if (!first)
				{
					out << ", ";
				}
				first = false;
				out << *it;
			}
			out << "]";
		}

		out << "}";
		return out.str();
	}

    std::string shape_string() const
    {
        std::ostringstream out;
        out << "[";
        for (size_t i = 0; i < dimensions_.size(); ++i)
        {
            if (i > 0)
            {
                out << ", ";
            }
            out << dimensions_[i];
        }
        out << "]";
        return out.str();
    }

private:
	void* root_ = nullptr;
	std::vector<int> dimensions_;
	MLCouplingMemoryLayout layout_ = MLCouplingMemLayoutInvalid;
	MLCouplingOwnership ownership_ = MLCouplingOwnershipExternal;
	std::function<void(void*)> deleter_;
	std::shared_ptr<void> owner_;

	static std::shared_ptr<void> make_owner(void* root,
											 MLCouplingMemoryLayout layout,
											 const std::function<void(void*)>& deleter)
	{
		if (root == nullptr)
		{
			return {};
		}
		if (deleter)
		{
			return std::shared_ptr<void>(root, deleter);
		}
		if (is_contiguous_layout(layout))
		{
			return std::shared_ptr<void>(root, [](void* ptr) { delete[] static_cast<T*>(ptr); });
		}
		return {};
	}

	static size_t numel_from_shape(const std::vector<int>& dimensions)
	{
		if (dimensions.empty())
		{
			return 0;
		}
		size_t total = 1;
		for (int d : dimensions)
		{
			if (d < 0)
			{
				throw std::invalid_argument("Negative dimension is invalid");
			}
			total *= static_cast<size_t>(d);
		}
		return total;
	}

	static std::vector<int> unravel_index(size_t linear_index,
										  const std::vector<int>& dimensions,
										  MLCouplingMemoryLayout layout)
	{
		std::vector<int> idx(dimensions.size(), 0);
		if (dimensions.empty())
		{
			return idx;
		}

		if (!is_fortran_layout(layout))
		{
			for (size_t i = dimensions.size(); i-- > 0;)
			{
				const size_t dim = static_cast<size_t>(dimensions[i]);
				idx[i] = static_cast<int>(linear_index % dim);
				linear_index /= dim;
			}
			return idx;
		}

		for (size_t i = 0; i < dimensions.size(); ++i)
		{
			const size_t dim = static_cast<size_t>(dimensions[i]);
			idx[i] = static_cast<int>(linear_index % dim);
			linear_index /= dim;
		}
		return idx;
	}

	static size_t linear_offset(const std::vector<int>& index,
								const std::vector<int>& dimensions,
								MLCouplingMemoryLayout layout)
	{
		if (index.size() != dimensions.size())
		{
			throw std::invalid_argument("linear_offset(): index rank mismatch");
		}

		size_t offset = 0;

		if (!is_fortran_layout(layout))
		{
			size_t stride = 1;
			for (size_t i = dimensions.size(); i-- > 0;)
			{
				offset += static_cast<size_t>(index[i]) * stride;
				stride *= static_cast<size_t>(dimensions[i]);
			}
			return offset;
		}

		size_t stride = 1;
		for (size_t i = 0; i < dimensions.size(); ++i)
		{
			offset += static_cast<size_t>(index[i]) * stride;
			stride *= static_cast<size_t>(dimensions[i]);
		}
		return offset;
	}

	static size_t stride_for_depth(const std::vector<int>& dimensions,
								   MLCouplingMemoryLayout layout,
								   size_t depth)
	{
		if (dimensions.empty() || depth >= dimensions.size())
		{
			return 0;
		}

		size_t stride = 1;

		if (!is_fortran_layout(layout))
		{
			for (size_t i = depth + 1; i < dimensions.size(); ++i)
			{
				stride *= static_cast<size_t>(dimensions[i]);
			}
			return stride;
		}

		for (size_t i = 0; i < depth; ++i)
		{
			stride *= static_cast<size_t>(dimensions[i]);
		}
		return stride;
	}

	static void free_nested_tree(void* root, const std::vector<int>& dimensions, size_t depth)
	{
		if (root == nullptr)
		{
			return;
		}

		if (dimensions.empty())
		{
			return;
		}

		if (dimensions.size() == 1)
		{
			delete[] static_cast<T*>(root);
			return;
		}

		if (depth + 1 == dimensions.size() - 1)
		{
			void** table = static_cast<void**>(root);
			for (int i = 0; i < dimensions[depth]; ++i)
			{
				delete[] static_cast<T*>(table[i]);
			}
			delete[] table;
			return;
		}

		void** table = static_cast<void**>(root);
		for (int i = 0; i < dimensions[depth]; ++i)
		{
			free_nested_tree(table[i], dimensions, depth + 1);
		}
		delete[] table;
	}

	void validate_shape() const
	{
		for (int d : dimensions_)
		{
			if (d < 0)
			{
				throw std::invalid_argument("Negative dimension is invalid");
			}
		}

		if (layout_ != MLCouplingMemLayoutInvalid && root_ == nullptr && numel() > 0)
		{
			throw std::invalid_argument("Non-empty tensor requires non-null root pointer");
		}
	}

	void validate_index(const std::vector<int>& index) const
	{
		if (index.size() != dimensions_.size())
		{
			throw std::invalid_argument("Index rank does not match tensor rank");
		}
		for (size_t i = 0; i < index.size(); ++i)
		{
			if (index[i] < 0 || index[i] >= dimensions_[i])
			{
				throw std::out_of_range("Tensor index out of bounds");
			}
		}
	}

	void release_owned()
	{
		owner_.reset();
		root_ = nullptr;
		ownership_ = MLCouplingOwnershipExternal;
		deleter_ = {};
	}

	void copy_from(const MLCouplingTensor& other)
	{
		dimensions_ = other.dimensions_;
		layout_ = other.layout_;

		if (other.root_ == nullptr || other.numel() == 0)
		{
			root_ = nullptr;
			ownership_ = MLCouplingOwnershipExternal;
			deleter_ = {};
			owner_.reset();
			return;
		}

		const MLCouplingTensor flat = other.flatten(to_contiguous_layout(other.layout_));

		if (is_contiguous_layout(other.layout_))
		{
			const size_t n = flat.numel();
			T* copied = new T[n];
			std::copy(static_cast<T*>(flat.root_), static_cast<T*>(flat.root_) + n, copied);
			root_ = static_cast<void*>(copied);
			ownership_ = MLCouplingOwnershipOwned;
			deleter_ = [](void* ptr) { delete[] static_cast<T*>(ptr); };
			owner_ = make_owner(root_, layout_, deleter_);
			return;
		}

		MLCouplingTensor nested_copy = flat.unflatten(other.layout_);
		root_ = nested_copy.root_;
		ownership_ = MLCouplingOwnershipOwned;
		deleter_ = std::move(nested_copy.deleter_);
		owner_ = std::move(nested_copy.owner_);

		nested_copy.root_ = nullptr;
		nested_copy.ownership_ = MLCouplingOwnershipExternal;
		nested_copy.deleter_ = {};
		nested_copy.owner_.reset();
	}
};

template <typename T>
class MLCouplingData
{
public:
	using container_type = std::vector<MLCouplingTensor<T>>;
	using iterator = typename container_type::iterator;
	using const_iterator = typename container_type::const_iterator;

	MLCouplingData() = default;

	explicit MLCouplingData(std::vector<MLCouplingTensor<T>> tensors)
		: tensors_(std::move(tensors)) {}

	void add_tensor(const MLCouplingTensor<T>& tensor)
	{
		tensors_.push_back(tensor);
	}

	void add_tensor(MLCouplingTensor<T>&& tensor)
	{
		tensors_.push_back(std::move(tensor));
	}

	size_t size() const { return tensors_.size(); }
	bool empty() const { return tensors_.empty(); }

	MLCouplingTensor<T>& operator[](size_t i) { return tensors_.at(i); }
	const MLCouplingTensor<T>& operator[](size_t i) const { return tensors_.at(i); }

	iterator begin() { return tensors_.begin(); }
	iterator end() { return tensors_.end(); }
	const_iterator begin() const { return tensors_.begin(); }
	const_iterator end() const { return tensors_.end(); }
	const_iterator cbegin() const { return tensors_.cbegin(); }
	const_iterator cend() const { return tensors_.cend(); }

	MLCouplingData<T> flatten_all(MLCouplingMemoryLayout target_layout = MLCouplingMemLayoutContiguous) const
	{
		MLCouplingData<T> out;
		out.tensors_.reserve(tensors_.size());
		for (const auto& t : tensors_)
		{
			out.tensors_.push_back(t.flatten(target_layout));
		}
		return out;
	}

	MLCouplingData<T> deep_copy() const
	{
		MLCouplingData<T> out;
		out.tensors_.reserve(tensors_.size());
		for (const auto& tensor : tensors_)
		{
			out.tensors_.push_back(tensor.deep_copy());
		}
		return out;
	}

	std::vector<T> get_flat_data() const
	{
		std::vector<T> combined;
		size_t total = 0;
		for (const auto& tensor : tensors_)
		{
			total += tensor.numel();
		}
		combined.reserve(total);

		for (const auto& tensor : tensors_)
		{
			std::vector<T> local = tensor.as_flat_vector(MLCouplingMemLayoutContiguous);
			combined.insert(combined.end(), local.begin(), local.end());
		}

		return combined;
	}

	std::string to_string(const std::string& class_name = "MLCouplingData") const
	{
		std::ostringstream out;
		out << class_name << "{num_tensors=" << tensors_.size() << ", tensors=[";
		for (size_t i = 0; i < tensors_.size(); ++i)
		{
			if (i > 0)
			{
				out << ", ";
			}
			out << tensors_[i].to_string("MLCouplingTensor");
		}
		out << "]}";
		return out.str();
	}

private:
	std::vector<MLCouplingTensor<T>> tensors_;
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const MLCouplingTensor<T>& tensor)
{
	os << tensor.to_string();
	return os;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const MLCouplingData<T>& data)
{
	os << data.to_string();
	return os;
}

template <typename T>
class MLCouplingScalar : public MLCouplingTensor<T>
{
public:
	explicit MLCouplingScalar(T value)
		: MLCouplingTensor<T>(MLCouplingTensor<T>::from_flat_copy(std::vector<T>{value}, std::vector<int>{1})) {}

	explicit MLCouplingScalar(T* ptr, MLCouplingOwnership ownership = MLCouplingOwnershipExternal)
		: MLCouplingTensor<T>(MLCouplingTensor<T>::wrap_flat(ptr, std::vector<int>{1}, MLCouplingMemLayoutContiguous, ownership)) {}
};

template <typename T>
class MLCouplingVector : public MLCouplingTensor<T>
{
public:
	explicit MLCouplingVector(std::vector<T> values)
		: MLCouplingTensor<T>(MLCouplingTensor<T>::from_flat_copy(values, std::vector<int>{static_cast<int>(values.size())})) {}

	MLCouplingVector(T* ptr, int size, MLCouplingOwnership ownership = MLCouplingOwnershipExternal)
		: MLCouplingTensor<T>(MLCouplingTensor<T>::wrap_flat(ptr, std::vector<int>{size}, MLCouplingMemLayoutContiguous, ownership)) {}
};

template <typename T>
class MLCouplingMatrix : public MLCouplingTensor<T>
{
public:
	MLCouplingMatrix(T* ptr, int rows, int cols, MLCouplingOwnership ownership = MLCouplingOwnershipExternal)
		: MLCouplingTensor<T>(MLCouplingTensor<T>::wrap_flat(ptr, std::vector<int>{rows, cols}, MLCouplingMemLayoutContiguous, ownership)) {}

	explicit MLCouplingMatrix(std::vector<std::vector<T>> values)
		: MLCouplingTensor<T>(make_from_nested_vector(std::move(values))) {}

private:
	static MLCouplingTensor<T> make_from_nested_vector(std::vector<std::vector<T>> values)
	{
		const int rows = static_cast<int>(values.size());
		const int cols = rows == 0 ? 0 : static_cast<int>(values[0].size());
		std::vector<T> flat;
		flat.reserve(static_cast<size_t>(rows) * static_cast<size_t>(cols));

		for (const auto& row : values)
		{
			if (static_cast<int>(row.size()) != cols)
			{
				throw std::invalid_argument("MLCouplingMatrix(): rows must have equal length");
			}
			flat.insert(flat.end(), row.begin(), row.end());
		}

		return MLCouplingTensor<T>::from_flat_copy(flat, std::vector<int>{rows, cols});
	}
};
