import unittest, os, logging
import random
import string
from itertools import islice
from math import ceil
from PIL import Image
from unittest import skip
from unittest.mock import Mock, MagicMock, patch
from helpers.multiaspect.sampler import MultiAspectSampler
from helpers.metadata.backends.discovery import DiscoveryMetadataBackend
from helpers.multiaspect.state import BucketStateManager
from tests.helpers.data import MockDataBackend
from accelerate import PartialState
from PIL import Image


class TestMultiAspectSampler(unittest.TestCase):
    def setUp(self):
        self.process_state = PartialState()
        self.accelerator = MagicMock()
        self.accelerator.log = MagicMock()
        self.metadata_backend = Mock(spec=DiscoveryMetadataBackend)
        self.metadata_backend.id = "foo"
        self.metadata_backend.aspect_ratio_bucket_indices = {
            "1.0": ["image1", "image2", "image3", "image4"],
        }
        self.metadata_backend.seen_images = {}
        self.metadata_backend.instance_data_dir = "foo"
        self.data_backend = MockDataBackend()
        self.data_backend.id = "foo"
        self.batch_size = 2
        self.seen_images_path = "/some/fake/seen_images.json"
        self.state_path = "/some/fake/state.json"

        self.sampler = MultiAspectSampler(
            id="foo",
            metadata_backend=self.metadata_backend,
            data_backend=self.data_backend,
            accelerator=self.accelerator,
            batch_size=self.batch_size,
            minimum_image_size=0,
            resolution_type="pixel",
            resolution=64,
        )

        self.sampler.state_manager = Mock(spec=BucketStateManager)
        self.sampler.state_manager.load_state.return_value = {}

        class FakeGetArgs:
            model_type = 'foo'
            aspect_bucket_rounding = 2
            aspect_bucket_alignment = 64  # Add missing attribute to mock StateTracker.get_args()

        # Consistent mock for StateTracker.get_args()
        patcher = patch('helpers.multiaspect.sampler.StateTracker.get_args', return_value=FakeGetArgs())
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_len(self):
        self.assertEqual(len(self.sampler), 2)

    def test_save_state(self):
        with patch.object(self.sampler.state_manager, "save_state") as mock_save_state:
            self.sampler.save_state(self.state_path)
        mock_save_state.assert_called_once()

    def test_load_buckets(self):
        buckets = self.sampler.load_buckets()
        self.assertEqual(buckets, ["1.0"])

    def test_change_bucket(self):
        self.sampler.buckets = ["1.5"]
        self.sampler.exhausted_buckets = ["1.0"]
        self.sampler.change_bucket()
        self.assertEqual(self.sampler.current_bucket, 0)  # Should now point to '1.5'

    def test_move_to_exhausted(self):
        self.sampler.current_bucket = 0  # Pointing to '1.0'
        self.sampler.buckets = ["1.0"]
        self.sampler.change_bucket()
        self.sampler.move_to_exhausted()
        self.assertEqual(self.sampler.exhausted_buckets, ["1.0"])
        self.assertEqual(self.sampler.buckets, [])

    @patch('helpers.multiaspect.sampler.MultiAspectSampler._reset_buckets')  # Prevent reset during test
    @patch('helpers.multiaspect.sampler.StateTracker.get_conditioning_dataset')  # Mock conditioning dataset retrieval
    def test_iter_yields_correct_batches(self, mock_get_conditioning_dataset, mock_reset_buckets):
        # Set up mock conditioning dataset
        mock_conditioning_sample = {'conditioning_sample': 'sample'}
        mock_get_conditioning_dataset.return_value = {'sampler': MagicMock(get_conditioning_sample=MagicMock(return_value=mock_conditioning_sample))}

        # Set up test data with 100 images in one bucket
        all_images = ["image" + str(i) for i in range(100)]
        batch_size = 4
        self.metadata_backend.aspect_ratio_bucket_indices = {"1.0": all_images}
        self.sampler.buckets = ["1.0"]
        self.sampler.exhausted_buckets = []  # Start with no exhausted buckets

        # Mock unseen images to limit returns after a set number of calls
        self.sampler._get_unseen_images = MagicMock(side_effect=lambda bucket: all_images[:batch_size] if bucket == "1.0" else [])

        # Mock exists check and image reading
        self.data_backend.exists = MagicMock(return_value=True)
        self.sampler._get_image_files = MagicMock(return_value=all_images)

        # Mock _validate_and_yield_images_from_samples to return tuples for dimensions
        self.sampler._validate_and_yield_images_from_samples = MagicMock(return_value=[{'image_path': img, 'target_size': (100, 100)} for img in all_images[:batch_size]])
        
        # Control iteration over batches to prevent infinite loop
        batches = []
        with patch("PIL.Image.open", return_value=MagicMock(spec=Image.Image)):
            for batch_item in islice(self.sampler, ceil(len(all_images) / batch_size)):
                batches.append(batch_item)

        # Check batch length consistency
        self.assertEqual(len(batches), ceil(len(all_images) / batch_size))

    @patch('helpers.multiaspect.sampler.MultiAspectSampler._reset_buckets')
    @patch('helpers.multiaspect.sampler.StateTracker.get_conditioning_dataset')  # Mock conditioning dataset retrieval
    def test_iter_handles_small_images(self, mock_get_conditioning_dataset, mock_reset_buckets):
        # Set up mock conditioning dataset
        mock_conditioning_sample = {'conditioning_sample': 'sample'}
        mock_get_conditioning_dataset.return_value = {'sampler': MagicMock(get_conditioning_sample=MagicMock(return_value=mock_conditioning_sample))}

        # Mock _validate_and_yield_images_from_samples to filter out small images
        def mock_validate_and_yield_images_from_samples(samples, bucket):
            # Return dictionary with "image_path" and "target_size" for valid images
            return [{'image_path': img, 'target_size': (100, 100)} for img in samples if img != "image2"]

        self.metadata_backend.aspect_ratio_bucket_indices = {
            "1.0": ["image1", "image2", "image3", "image4"]
        }
        self.sampler.buckets = ["1.0"]
        self.sampler.exhausted_buckets = []  # Prevent all buckets from exhausting
        self.sampler._validate_and_yield_images_from_samples = mock_validate_and_yield_images_from_samples
        
        # Mock _get_unseen_images to return a subset and avoid ZeroDivisionError
        self.sampler._get_unseen_images = MagicMock(return_value=["image1", "image3", "image4"])

        # Mock random.sample to control the order of images in batches, cycling through batches if more are requested
        def cycle_samples(*args, **kwargs):
            if args[0] == ["image1", "image3", "image4"]:
                return ["image1", "image3"]
            return ["image4"]

        with patch('random.sample', side_effect=cycle_samples):
            # Limit batches to prevent infinite looping
            batches = list(islice(self.sampler, 2))

        # Assert batch contents
        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0], (
            {'image_path': 'image1', 'target_size': (100, 100)},
            {'image_path': 'image3', 'target_size': (100, 100)},
            {'conditioning_sample': 'sample'},
            {'conditioning_sample': 'sample'},
        ))

    def test_iter_handles_incorrect_aspect_ratios_with_real_logic(self):
        try:
            # Setup a temporary directory for test images
            tmp_path = ''.join(random.choices(string.ascii_lowercase + string.digits, k=16))
            os.makedirs(tmp_path)

            # Define image paths and create images
            img_paths = [
                os.path.join(tmp_path, "image1.jpg"),
                os.path.join(tmp_path, "image2.jpg"),
                os.path.join(tmp_path, "incorrect_image.jpg"),
                os.path.join(tmp_path, "image4.jpg"),
            ]
            Image.new("RGB", (100, 100), color="red").save(img_paths[0])
            Image.new("RGB", (100, 100), color="green").save(img_paths[1])
            Image.new("RGB", (50, 100), color="blue").save(img_paths[2])  # Different aspect ratio
            Image.new("RGB", (100, 100), color="yellow").save(img_paths[3])

            # Set up the sampler metadata and bucket indices
            self.sampler.metadata_backend.instance_data_dir = tmp_path
            relative_img_paths = [os.path.basename(p) for p in img_paths]
            self.sampler.metadata_backend.aspect_ratio_bucket_indices = {"1.0": relative_img_paths}

            # Mock seen_images behavior
            self.sampler.metadata_backend.seen_images = {}
            self.sampler.metadata_backend.is_seen = lambda image: self.sampler.metadata_backend.seen_images.get(image, False)
            self.sampler.metadata_backend.mark_batch_as_seen = lambda images: self.sampler.metadata_backend.seen_images.update({os.path.basename(image): True for image in images})

            # Mock read_image to open images from filesystem
            self.sampler.data_backend.read_image = lambda path: Image.open(path)

            # Mock get_metadata_by_filepath to return target sizes
            def mock_get_metadata_by_filepath(path):
                target_size = (50, 100) if os.path.basename(path) == "incorrect_image.jpg" else (100, 100)
                return {
                    "image_path": path,
                    "data_backend_id": self.sampler.id,
                    "target_size": target_size
                }
            self.sampler.metadata_backend.get_metadata_by_filepath = mock_get_metadata_by_filepath

            # Mock StateTracker methods
            with patch('helpers.multiaspect.sampler.StateTracker') as MockStateTracker:
                MockStateTracker.get_args.return_value = Mock()
                MockStateTracker.get_args.return_value.aspect_bucket_rounding = 2
                MockStateTracker.get_args.return_value.model_type = 'legacy'
                MockStateTracker.get_data_backend_config.return_value = {'repeats': 0}

                # Ensure that get_conditioning_dataset returns None to avoid conditioning sample addition
                MockStateTracker.get_conditioning_dataset.return_value = None

                # Collect batches
                batches = []
                sampler_iter = iter(self.sampler)
                for _ in range(len(relative_img_paths) // self.sampler.batch_size):
                    batch = next(sampler_iter)
                    batches.append(batch)

                # Collect all image paths from batches
                all_image_paths_in_batches = []
                for batch in batches:
                    for item in batch:
                        image_path = item["image_path"]
                        all_image_paths_in_batches.append(image_path)

                incorrect_image_full_path = os.path.join(tmp_path, 'incorrect_image.jpg')

                # Assert that the incorrect image is not in the batches
                self.assertNotIn(incorrect_image_full_path, all_image_paths_in_batches)

                # Check that all images have the same size (100x100)
                first_image = self.sampler.data_backend.read_image(all_image_paths_in_batches[0])
                first_img_size = first_image.size
                for image_path in all_image_paths_in_batches:
                    image = self.sampler.data_backend.read_image(image_path)
                    self.assertEqual(image.size, first_img_size)

        finally:
            # Clean up the test images and directory
            for img_path in img_paths:
                os.remove(img_path)
            os.rmdir(tmp_path)

    def test_save_state_calls_state_manager(self):
        state_path = "/fake/path/state.json"
        with patch.object(self.sampler.state_manager, 'save_state') as mock_save_state:
            self.sampler.save_state(state_path)
        mock_save_state.assert_called_once()
        args, kwargs = mock_save_state.call_args
        self.assertEqual(args[1], state_path)

    def test_load_states_restores_state(self):
        state_path = "/fake/path/state.json"
        previous_state = {
            'aspect_ratio_bucket_indices': {'1.0': ['image1', 'image2']},
            'buckets': ['1.0'],
            'exhausted_buckets': ['2.0'],
            'batch_size': 2,
            'current_bucket': 0,
            'seen_images': {'image1': True},
            'current_epoch': 2,
        }
        self.sampler.state_manager.load_state.return_value = previous_state
        self.sampler.load_states(state_path)
        self.assertEqual(self.sampler.buckets, ['1.0'])
        self.assertEqual(self.sampler.exhausted_buckets, ['2.0'])
        self.assertEqual(self.sampler.current_epoch, 2)
        self.assertEqual(self.metadata_backend.seen_images, {'image1': True})

    def test_get_unseen_images(self):
        self.metadata_backend.aspect_ratio_bucket_indices = {
            '1.0': ['image1', 'image2', 'image3'],
            '1.5': ['image4', 'image5'],
        }
        self.metadata_backend.is_seen = lambda image: image == 'image2'
        unseen_images = self.sampler._get_unseen_images('1.0')
        self.assertEqual(unseen_images, [os.path.join(self.metadata_backend.instance_data_dir, 'image1'), os.path.join(self.metadata_backend.instance_data_dir, 'image3')])

    def test_handle_bucket_with_insufficient_images(self):
        self.sampler.buckets = ['1.0', '1.5']
        self.sampler.current_bucket = 0
        self.metadata_backend.aspect_ratio_bucket_indices = {
            '1.0': ['image1'],
            '1.5': ['image2', 'image3'],
        }
        self.metadata_backend.seen_images = {}
        result = self.sampler._handle_bucket_with_insufficient_images('1.0')
        self.assertTrue(result)
        self.assertEqual(self.sampler.exhausted_buckets, ['1.0'])
        self.assertEqual(self.sampler.buckets, ['1.5'])

    def test_get_next_bucket(self):
        # Set up initial conditions
        self.sampler.buckets = ['1.0', '1.5', '2.0']
        self.sampler.exhausted_buckets = ['1.5']
        self.sampler.current_bucket = 0

        # Patch random.choice to return '2.0' deterministically when called
        with patch('random.choice', return_value='2.0'):
            next_bucket = self.sampler._get_next_bucket()
            self.assertEqual(next_bucket, '2.0')  # Expect '2.0' due to mock

    def test_change_bucket(self):
        self.sampler.buckets = ['1.0', '1.5', '2.0']
        self.sampler.exhausted_buckets = []
        self.sampler.current_bucket = None
        self.sampler.change_bucket()
        self.assertEqual(self.sampler.current_bucket, 0)
        self.sampler.change_bucket()
        self.assertEqual(self.sampler.current_bucket, 1)

    def test_move_to_exhausted(self):
        self.sampler.buckets = ['1.0', '1.5', '2.0']
        self.sampler.current_bucket = 1  # '1.5'
        self.sampler.move_to_exhausted()
        self.assertEqual(self.sampler.exhausted_buckets, ['1.5'])
        self.assertEqual(self.sampler.buckets, ['1.0', '2.0'])

    def test_clear_batch_accumulator(self):
        self.sampler.batch_accumulator = ['image1', 'image2']
        self.sampler._clear_batch_accumulator()
        self.assertEqual(self.sampler.batch_accumulator, [])

    def test_validate_and_yield_images_from_samples(self):
        samples = ['image1', 'image2']
        bucket = '1.0'
        self.metadata_backend.get_metadata_by_filepath = lambda path: {'image_path': path, 'crop_coordinates': (0,0)}

        with (
            patch('helpers.multiaspect.sampler.PromptHandler.magic_prompt', return_value='A test prompt'),
        ):
            to_yield = self.sampler._validate_and_yield_images_from_samples(samples, bucket)
            self.assertEqual(len(to_yield), 2)
            self.assertEqual(to_yield[0]['image_path'], 'image1')
            self.assertEqual(to_yield[0]['instance_prompt_text'], 'A test prompt')

    def test_retrieve_validation_set(self):
        # Mock methods and properties
        self.sampler._yield_random_image = MagicMock(side_effect=['image1', 'image2'])
        self.sampler.data_backend.read_image = MagicMock(
            return_value=Image.new("RGB", (100, 100), color="red")
        )
        self.sampler.metadata_backend.get_metadata_by_filepath = lambda path: {'image_path': path}
        
        # Mock StateTracker configuration
        with (
            patch('helpers.multiaspect.sampler.PromptHandler.magic_prompt', return_value='A test prompt'),
            patch('helpers.multiaspect.sampler.StateTracker.get_data_backend_config', return_value={
                'resolution_type': 'pixel',
                'crop': False,
                'crop_style': 'random',
                'resolution': 64,
                'size_bucket_increment': 64,
                'maximum_image_size': 1024,
                'minimum_image_size': 512,
            })
        ):
            results = self.sampler.retrieve_validation_set(2)
        
        # Verify results
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], 'foo_0')
        self.assertEqual(results[0][1], 'A test prompt')
        self.assertIsInstance(results[0][2], Image.Image)  # Verify the image type

    def test_get_conditioning_sample(self):
        # Create a mock image with PIL that has a size attribute
        mock_image = Image.new("RGB", (100, 100))  # A 100x100 image

        # Set up the data_backend to return the mock image
        self.sampler.data_backend.read_image = MagicMock(return_value=mock_image)

        # Mock the metadata_backend to return sample metadata
        self.sampler.metadata_backend.get_metadata_by_filepath = lambda path: {'image_path': path}

        # Assume instance_data_dir is set up for full path construction
        self.sampler.metadata_backend.instance_data_dir = "/mock/dir"

        # Call get_conditioning_sample and verify the output
        with (
            patch('helpers.multiaspect.sampler.PromptHandler.magic_prompt', return_value='A test prompt'),
            patch('helpers.multiaspect.sampler.StateTracker.get_data_backend_config', return_value={
                'resolution_type': 'pixel',
                'crop': False,
                'crop_style': 'random',
                'resolution': 64,
                'size_bucket_increment': 64,
                'maximum_image_size': 1024,
                'minimum_image_size': 512,
            })
        ):
            sample = self.sampler.get_conditioning_sample('image1')
        self.assertEqual(sample.image, mock_image)
        self.assertEqual(
            sample.image_metadata['image_path'],
            os.path.join(self.sampler.metadata_backend.instance_data_dir, 'image1')
        )

    def test_connect_conditioning_samples(self):
        # Set instance_data_dir to a string, not a MagicMock
        self.metadata_backend.instance_data_dir = 'foo'
        
        # Define test samples and mocked conditioning dataset
        samples = ({'image_path': 'image1'}, {'image_path': 'image2'})
        conditioning_dataset = {'sampler': MagicMock()}
        conditioning_sample = 'conditioning_sample'
        conditioning_dataset['sampler'].get_conditioning_sample.return_value = conditioning_sample

        # Patch StateTracker to return the conditioning dataset mock
        with patch('helpers.multiaspect.sampler.StateTracker.get_conditioning_dataset', return_value=conditioning_dataset):
            outputs = self.sampler.connect_conditioning_samples(samples)

        # Modify assertion to match the output's type (tuple)
        self.assertEqual(outputs, tuple(list(samples) + [conditioning_sample, conditioning_sample]))

    def test_convert_to_human_readable(self):
        res = MultiAspectSampler.convert_to_human_readable(1.0, ['image1', 'image2'], resolution=512)
        self.assertEqual(res, '1.0 (2 samples)')

if __name__ == "__main__":
    unittest.main()
