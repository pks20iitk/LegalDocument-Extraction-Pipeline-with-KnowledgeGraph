
# Create your tests here.
from django.test import TestCase
from data import user_data  # Import the user_data module

class TestContractType(TestCase):
    def test_set_question_generation_prompt(self):
        # Define a custom prompt template for testing
        custom_prompt_template = "Your custom prompt template goes here"

        # Call the function and get the prompt
        prompt = set_question_generation_prompt()

        # Check if the prompt is an instance of PromptTemplate
        self.assertIsInstance(prompt, PromptTemplate)

        # Check if the template attribute of the PromptTemplate matches the custom_prompt_template
        self.assertEqual(prompt.template, custom_prompt_template)

        # Check if the input_variables attribute of the PromptTemplate matches the expected list
        self.assertListEqual(prompt.input_variables, ['context', 'question'])
    def setUp(self):
        # Initialize the test client
        self.client = APIClient()

    def test_getContractType(self):
        # Define a sample URL to test
        sample_url = "https://legalgraphaiprod.blob.core.windows.net/cached-apps/05 - MSA_US-linkedin.pdf"

        # Define the expected response data
        expected_response = {
            "doc_type": "Sample Document Type"  # Replace with the expected type
        }

        # Make a POST request to the view with the sample URL
        response = self.client.post(reverse("your_app:get-contract-type"), {"url": sample_url})

        # Check if the response status code is 200 (OK)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        # Check if the response data matches the expected response
        self.assertEqual(response.data, expected_response)
    @patch('ragService.views.process_url_type')
    @patch('ragService.views.get_type_of_contract')
    @patch('ragService.views.AzureChatOpenAI')
    def test_getContractType(self, mock_azure_class, mock_get_type_of_contract, mock_process_url_type):
        # Mock the dependencies to isolate the function
        mock_response = MagicMock()
        mock_response.data = "Sample Document Type"
        mock_process_url_type.return_value = True
        mock_get_type_of_contract.return_value = mock_response

        # Define a sample URL to test
        sample_url = "https://legalgraphaiprod.blob.core.windows.net/cached-apps/05 - MSA_US-linkedin.pdf"

        # Make a POST request to the view with the sample URL
        response = self.client.post(reverse("your_app:get-contract-type"), {"url": sample_url})

        # Check if the response status code is 200 (OK)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        # Check if the response data matches the expected response
        self.assertEqual(response.data, {"doc_type": "Sample Document Type"})

        # Ensure that process_url_type and get_type_of_contract were called with the correct arguments
        mock_process_url_type.assert_called_once_with(sample_url)
        mock_get_type_of_contract.assert_called_once()

        # Ensure that AzureChatOpenAI was instantiated with the correct arguments
        mock_azure_class.assert_called_once_with(
            temperature=0,
            openai_api_type="azure",
            openai_api_key=openai_key,
            openai_api_base=openai_base,
            deployment_name=openai_chat_deployment_name,
            model="gpt-3.5-turbo-16k",
            openai_api_version="2023-05-15",
        )

    

# TODO: Test with tables