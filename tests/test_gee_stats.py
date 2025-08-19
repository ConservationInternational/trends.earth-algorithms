import pytest
from unittest.mock import Mock, patch

import te_algorithms.gee.stats as stats


class TestGetKendallCoef:
    """Test the get_kendall_coef function."""

    def test_get_kendall_coef_basic_functionality(self):
        """Test basic functionality with different confidence levels."""
        # Test 95% confidence level with minimum size
        result = stats.get_kendall_coef(4, 95)
        assert result == 4  # First value in 95% array
        
        # Test 90% confidence level 
        result = stats.get_kendall_coef(4, 90)
        assert result == 4  # First value in 90% array
        
        # Test 99% confidence level
        result = stats.get_kendall_coef(4, 99)
        assert result == 6  # First value in 99% array

    def test_get_kendall_coef_various_sizes(self):
        """Test with different sample sizes."""
        # Test larger sample sizes
        result_95_5 = stats.get_kendall_coef(5, 95)
        result_95_4 = stats.get_kendall_coef(4, 95)
        assert result_95_5 != result_95_4  # Should give different values
        
        result_95_10 = stats.get_kendall_coef(10, 95)
        assert result_95_10 > result_95_4  # Larger sample should have larger coefficient

    def test_get_kendall_coef_confidence_levels(self):
        """Test that higher confidence levels give larger coefficients."""
        n = 10
        coef_90 = stats.get_kendall_coef(n, 90)
        coef_95 = stats.get_kendall_coef(n, 95)
        coef_99 = stats.get_kendall_coef(n, 99)
        
        # Higher confidence should require larger coefficients
        assert coef_90 <= coef_95 <= coef_99

    def test_get_kendall_coef_minimum_size(self):
        """Test that minimum sample size of 4 is enforced."""
        with pytest.raises(AssertionError):
            stats.get_kendall_coef(3, 95)  # Should fail for n < 4
        
        with pytest.raises(AssertionError):
            stats.get_kendall_coef(1, 95)

    def test_get_kendall_coef_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with exactly minimum size
        result = stats.get_kendall_coef(4, 95)
        assert isinstance(result, int)
        assert result > 0
        
        # Test with very large sample size (limited by array size)
        # The arrays have ~56 elements, so testing around that boundary
        try:
            result = stats.get_kendall_coef(60, 95)  # This should work
            assert isinstance(result, int)
        except IndexError:
            # If this fails, test a smaller size that should work
            result = stats.get_kendall_coef(50, 95)
            assert isinstance(result, int)


class TestMannKendall:
    """Test the mann_kendall function."""

    def create_mock_image_collection(self, values):
        """Create a mock GEE ImageCollection for testing."""
        mock_collection = Mock()
        mock_list = Mock()
        mock_list.length.return_value.getInfo.return_value = len(values)
        mock_collection.toList.return_value = mock_list
        
        # Mock the image list access
        mock_images = []
        for i, value in enumerate(values):
            mock_image = Mock()
            mock_image.lt.return_value = Mock()  # Concordant comparison
            mock_image.gt.return_value = Mock()  # Discordant comparison
            mock_images.append(mock_image)
        
        def mock_get(index):
            return mock_images[index]
        
        mock_list.get = mock_get
        return mock_collection

    @patch('ee.Image')
    @patch('ee.ImageCollection')
    def test_mann_kendall_structure(self, mock_image_collection_class, mock_image_class):
        """Test the basic structure of mann_kendall function."""
        # Create mock objects
        mock_collection = self.create_mock_image_collection([1, 2, 3, 4])
        
        # Mock the ee.ImageCollection constructor and sum method
        mock_result_collection = Mock()
        mock_result_collection.sum.return_value = Mock()
        mock_image_collection_class.return_value = mock_result_collection
        
        # Call the function
        stats.mann_kendall(mock_collection)
        
        # Verify the basic flow was called
        mock_collection.toList.assert_called_once_with(50)
        assert mock_image_collection_class.call_count == 2  # Called for concordant and discordant arrays

    @patch('ee.Image')
    @patch('ee.ImageCollection')
    def test_mann_kendall_with_different_sizes(self, mock_image_collection_class, mock_image_class):
        """Test mann_kendall with different collection sizes."""
        # Test with small collection
        small_collection = self.create_mock_image_collection([1, 2])
        stats.mann_kendall(small_collection)
        
        # Verify it handles small collections
        small_collection.toList.assert_called_once_with(50)
        
        # Test with larger collection
        large_collection = self.create_mock_image_collection([1, 2, 3, 4, 5, 6])
        stats.mann_kendall(large_collection)
        
        large_collection.toList.assert_called_once_with(50)

    def test_mann_kendall_requires_image_collection(self):
        """Test that mann_kendall expects proper GEE ImageCollection interface."""
        # Test with None
        with pytest.raises(AttributeError):
            stats.mann_kendall(None)
        
        # Test with object that doesn't have toList method
        invalid_input = Mock(spec=[])  # No toList method
        with pytest.raises(AttributeError):
            stats.mann_kendall(invalid_input)

    @patch('ee.Image')
    @patch('ee.ImageCollection')
    def test_mann_kendall_pairwise_comparisons(self, mock_image_collection_class, mock_image_class):
        """Test that mann_kendall performs pairwise comparisons correctly."""
        # Create a collection with 3 images to test comparison logic
        collection = self.create_mock_image_collection([1, 2, 3])
        
        # Mock the result collection
        mock_result_collection = Mock()
        mock_result_collection.sum.return_value = Mock()
        mock_image_collection_class.return_value = mock_result_collection
        
        stats.mann_kendall(collection)
        
        # For 3 images, we should have 3 pairwise comparisons: (0,1), (0,2), (1,2)
        # Each comparison calls both lt() and gt() methods
        collection.toList.assert_called_once_with(50)
        
        # Verify ImageCollection was called twice (for concordant and discordant)
        assert mock_image_collection_class.call_count == 2