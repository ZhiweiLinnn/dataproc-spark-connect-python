# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from copy import deepcopy
import datetime
import os
import unittest

from google.api_core.exceptions import (
    Aborted,
    FailedPrecondition,
    InvalidArgument,
    NotFound,
)
from google.cloud.dataproc_spark_connect import DataprocSparkSession
from google.cloud.dataproc_spark_connect.exceptions import DataprocSparkConnectException
from google.cloud.dataproc_spark_connect.session import _is_valid_label_value, _is_valid_session_id
from google.cloud.dataproc_v1 import (
    AuthenticationConfig,
    CreateSessionRequest,
    GetSessionRequest,
    Session,
    SparkConnectConfig,
    TerminateSessionRequest,
)

from pyspark.sql.connect.client.core import ConfigResult
from pyspark.sql.connect.proto import Command, ConfigResponse, ExecutePlanRequest, Plan, SqlCommand, UserContext
from unittest import mock


class DataprocRemoteSparkSessionBuilderTests(unittest.TestCase):

    def setUp(self):
        self._default_runtime_version = (
            DataprocSparkSession._DEFAULT_RUNTIME_VERSION
        )
        self.original_environment = dict(os.environ)
        os.environ.clear()
        os.environ["GOOGLE_CLOUD_PROJECT"] = "test-project"
        os.environ["GOOGLE_CLOUD_REGION"] = "test-region"

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.original_environment)

    @staticmethod
    def stopSession(mock_session_controller_client_instance, session):
        session_response = Session()
        session_response.state = Session.State.TERMINATING
        mock_session_controller_client_instance.get_session.return_value = (
            session_response
        )
        if session is not None:
            session.stop()

    @staticmethod
    def _setup_session_creation_mocks(
        mock_is_s8s_session_active,
        mock_dataproc_session_id,
        mock_client_config,
        mock_session_controller_client,
        mock_credentials,
        session_id="sc-20240702-103952-abcdef",
        session_uuid="c002e4ef-fe5e-41a8-a157-160aa73e4f7f",
    ):
        """Helper method to set up common mocks for session creation tests."""
        mock_is_s8s_session_active.return_value = True
        mock_session_controller_client_instance = (
            mock_session_controller_client.return_value
        )
        mock_dataproc_session_id.return_value = session_id
        mock_client_config.return_value = ConfigResult.fromProto(
            ConfigResponse()
        )

        cred = mock.MagicMock()
        cred.token = "token"
        mock_credentials.return_value = (cred, "")

        mock_operation = mock.Mock()
        session_response = Session()
        session_response.runtime_info.endpoints = {
            "Spark Connect Server": "sc://spark-connect-server.example.com:443"
        }
        session_response.uuid = session_uuid
        mock_operation.result.side_effect = [session_response]
        mock_session_controller_client_instance.create_session.return_value = (
            mock_operation
        )

        return mock_session_controller_client_instance

    @mock.patch("google.auth.default")
    @mock.patch("google.cloud.dataproc_v1.SessionControllerClient")
    @mock.patch("pyspark.sql.connect.client.SparkConnectClient.config")
    @mock.patch(
        "google.cloud.dataproc_spark_connect.DataprocSparkSession.Builder.generate_dataproc_session_id"
    )
    @mock.patch(
        "google.cloud.dataproc_spark_connect.session.is_s8s_session_active"
    )
    @mock.patch(
        "google.cloud.dataproc_spark_connect.environment.get_client_environment_label"
    )
    @mock.patch(
        "IPython.core.interactiveshell.InteractiveShell.initialized",
        return_value=True,
    )
    @mock.patch.dict(
        "sys.modules",
        {
            "google.cloud.aiplatform.utils": mock.MagicMock(
                _ipython_utils=mock.MagicMock()
            ),
        },
    )
    def test_create_spark_session_with_default_notebook_behavior(
        self,
        mock_interactive_shell,
        mock_get_client_environment_label,
        mock_is_s8s_session_active,
        mock_dataproc_session_id,
        mock_client_config,
        mock_session_controller_client,
        mock_credentials,
    ):
        session = None
        mock_is_s8s_session_active.return_value = False  # No existing session
        mock_session_controller_client_instance = (
            mock_session_controller_client.return_value
        )

        session_id = "sc-20240702-103952-abcdef"
        mock_dataproc_session_id.return_value = session_id
        mock_client_config.return_value = ConfigResult.fromProto(
            ConfigResponse()
        )
        cred = mock.MagicMock()
        cred.token = "token"
        mock_credentials.return_value = (cred, "")
        mock_operation = mock.Mock()
        session_response = Session()
        session_response.runtime_info.endpoints = {
            "Spark Connect Server": "sc://spark-connect-server.example.com:443"
        }
        session_response.uuid = "c002e4ef-fe5e-41a8-a157-160aa73e4f7f"
        mock_operation.result.side_effect = [session_response]
        mock_session_controller_client_instance.create_session.return_value = (
            mock_operation
        )
        mock_get_client_environment_label.return_value = "unknown"
        mock_ipython_utils = mock.sys.modules[
            "google.cloud.aiplatform.utils"
        ]._ipython_utils
        test_session_url = f"https://console.cloud.google.com/dataproc/interactive/sessions/{session_id}/locations/test-region?project=test-project"
        mock_display_link = mock_ipython_utils.display_link
        mock.patch.dict(
            os.environ,
            {
                "VERTEX_PRODUCT": "COLAB_ENTERPRISE",
            },
        ).start()

        create_session_request = CreateSessionRequest()
        create_session_request.parent = (
            "projects/test-project/locations/test-region"
        )
        create_session_request.session.name = "projects/test-project/locations/test-region/sessions/sc-20240702-103952-abcdef"
        create_session_request.session.runtime_config.version = (
            self._default_runtime_version
        )
        create_session_request.session.spark_connect_session = (
            SparkConnectConfig()
        )
        create_session_request.session_id = "sc-20240702-103952-abcdef"
        create_session_request.session.labels["dataproc-session-client"] = (
            "unknown"
        )
        try:
            session = (
                DataprocSparkSession.builder.projectId("test-project")
                .location("test-region")
                .getOrCreate()
            )
            mock_session_controller_client_instance.create_session.assert_called_once_with(
                create_session_request
            )
        finally:
            mock_session_controller_client_instance.terminate_session.return_value = (
                mock.Mock()
            )
            self.stopSession(mock_session_controller_client_instance, session)
            terminate_session_request = TerminateSessionRequest()
            terminate_session_request.name = "projects/test-project/locations/test-region/sessions/sc-20240702-103952-abcdef"
            get_session_request = GetSessionRequest()
            get_session_request.name = "projects/test-project/locations/test-region/sessions/sc-20240702-103952-abcdef"
            mock_session_controller_client_instance.terminate_session.assert_called_once_with(
                terminate_session_request
            )
            mock_session_controller_client_instance.get_session.assert_called_once_with(
                get_session_request
            )
            mock_display_link.assert_called_once_with(
                "View Session Details", test_session_url, "dashboard"
            )

    @mock.patch("google.cloud.dataproc_v1.SessionControllerClient")
    def test_pypi_add_artifacts(
        self,
        mock_session_controller_client,
    ):
        session = None
        mock_session_controller_client_instance = (
            mock_session_controller_client.return_value
        )
        try:
            mock_operation = mock.Mock()
            session_response = Session()
            session_response.runtime_info.endpoints = {
                "Spark Connect Server": "sc://spark-connect-server.example.com:443"
            }
            session_response.uuid = "c002e4ef-fe5e-41a8-a157-160aa73e4f7f"
            mock_operation.result.side_effect = [session_response]
            mock_session_controller_client_instance.create_session.return_value = (
                mock_operation
            )
            session = DataprocSparkSession.builder.getOrCreate()
            self.assertTrue(isinstance(session, DataprocSparkSession))
            session.addArtifact = mock.MagicMock()

            # Setting two flags together
            with self.assertRaisesRegex(
                ValueError,
                "'pyfile', 'archive', 'file' and/or 'pypi' cannot be True together",
            ):
                session.addArtifacts("abc.txt", file=True, pypi=True)

            # Propagate error
            session.addArtifact.side_effect = Exception("Error installing")
            with self.assertRaisesRegex(
                Exception,
                "Error installing",
            ):
                session.addArtifacts("spacy", pypi=True)
            session.addArtifact.side_effect = None

            # Do install if earlier add artifact resulted in failure
            session.addArtifacts("spacy", pypi=True)
            self.assertEqual(session.addArtifact.call_count, 2)

            # test multiple packages, when already installed
            session.addArtifacts("spacy==1.2.3", "spacy", pypi=True)
            self.assertEqual(session.addArtifact.call_count, 3)
        finally:
            self.stopSession(mock_session_controller_client_instance, session)

    @mock.patch("google.auth.default")
    @mock.patch("google.cloud.dataproc_v1.SessionControllerClient")
    @mock.patch("pyspark.sql.connect.client.SparkConnectClient.config")
    @mock.patch(
        "google.cloud.dataproc_spark_connect.DataprocSparkSession.Builder.generate_dataproc_session_id"
    )
    @mock.patch(
        "google.cloud.dataproc_spark_connect.session.is_s8s_session_active"
    )
    def test_create_session_with_user_provided_dataproc_config(
        self,
        mock_is_s8s_session_active,
        mock_dataproc_session_id,
        mock_client_config,
        mock_session_controller_client,
        mock_credentials,
    ):
        session = None
        mock_is_s8s_session_active.return_value = True
        mock_session_controller_client_instance = (
            mock_session_controller_client.return_value
        )
        mock_client_config.return_value = ConfigResult.fromProto(
            ConfigResponse()
        )
        mock_dataproc_session_id.return_value = "sc-20240702-103952-abcdef"
        cred = mock.MagicMock()
        cred.token = "token"
        mock_credentials.return_value = (cred, "")
        mock_operation = mock.Mock()
        session_response = Session()
        session_response.runtime_info.endpoints = {
            "Spark Connect Server": "sc://spark-connect-server.example.com:443"
        }
        session_response.uuid = "c002e4ef-fe5e-41a8-a157-160aa73e4f7f"
        mock_operation.result.side_effect = [session_response]
        mock_session_controller_client_instance.create_session.return_value = (
            mock_operation
        )

        create_session_request = CreateSessionRequest()
        create_session_request.session.environment_config.execution_config.subnetwork_uri = (
            "user_passed_subnetwork_uri"
        )
        create_session_request.session.environment_config.execution_config.ttl = {
            "seconds": 10
        }
        create_session_request.parent = (
            "projects/test-project/locations/test-region"
        )
        create_session_request.session.name = "projects/test-project/locations/test-region/sessions/sc-20240702-103952-abcdef"
        create_session_request.session.runtime_config.properties = {
            "spark.executor.cores": "16"
        }
        create_session_request.session.runtime_config.version = (
            self._default_runtime_version
        )
        create_session_request.session.spark_connect_session = (
            SparkConnectConfig()
        )
        create_session_request.session_id = "sc-20240702-103952-abcdef"
        create_session_request.session.labels["dataproc-session-client"] = (
            "unknown"
        )

        try:
            dataproc_config = Session()
            dataproc_config.environment_config.execution_config.subnetwork_uri = (
                "user_passed_subnetwork_uri"
            )
            dataproc_config.environment_config.execution_config.ttl = {
                "seconds": 10
            }
            dataproc_config.runtime_config.properties = {
                "spark.executor.cores": "8"
            }
            session = (
                DataprocSparkSession.builder.config("spark.executor.cores", "6")
                .dataprocSessionConfig(dataproc_config)
                .config("spark.executor.cores", "16")
                .getOrCreate()
            )
            mock_session_controller_client_instance.create_session.assert_called_once_with(
                create_session_request
            )
        finally:
            mock_session_controller_client_instance.terminate_session.return_value = (
                mock.Mock()
            )
            self.stopSession(mock_session_controller_client_instance, session)
            terminate_session_request = TerminateSessionRequest()
            terminate_session_request.name = "projects/test-project/locations/test-region/sessions/sc-20240702-103952-abcdef"
            get_session_request = GetSessionRequest()
            get_session_request.name = "projects/test-project/locations/test-region/sessions/sc-20240702-103952-abcdef"
            mock_session_controller_client_instance.terminate_session.assert_called_once_with(
                terminate_session_request
            )
            mock_session_controller_client_instance.get_session.assert_called_once_with(
                get_session_request
            )

    @mock.patch("google.auth.default")
    @mock.patch("google.cloud.dataproc_v1.SessionControllerClient")
    @mock.patch(
        "google.cloud.dataproc_spark_connect.DataprocSparkSession.Builder.generate_dataproc_session_id"
    )
    @mock.patch(
        "google.cloud.dataproc_spark_connect.session.is_s8s_session_active"
    )
    def test_create_session_with_env_vars_config(
        self,
        mock_is_s8s_session_active,
        mock_dataproc_session_id,
        mock_session_controller_client,
        mock_credentials,
    ):
        session = None
        mock_is_s8s_session_active.return_value = True
        mock_session_controller_client_instance = (
            mock_session_controller_client.return_value
        )
        mock_dataproc_session_id.return_value = "sc-20240702-103952-abcdef"
        cred = mock.MagicMock()
        cred.token = "token"
        mock_credentials.return_value = (cred, "")
        mock_operation = mock.Mock()
        session_response = Session()
        session_response.runtime_info.endpoints = {
            "Spark Connect Server": "sc://spark-connect-server.example.com:443"
        }
        session_response.uuid = "c002e4ef-fe5e-41a8-a157-160aa73e4f7f"
        mock_operation.result.side_effect = [session_response]
        mock_session_controller_client_instance.create_session.return_value = (
            mock_operation
        )

        mock.patch.dict(
            os.environ,
            {
                "DATAPROC_SPARK_CONNECT_AUTH_TYPE": "SERVICE_ACCOUNT",
                "DATAPROC_SPARK_CONNECT_SERVICE_ACCOUNT": "test-acc@example.com",
                "DATAPROC_SPARK_CONNECT_SUBNET": "test-subnet-from-env",
                "DATAPROC_SPARK_CONNECT_TTL_SECONDS": "12",
                "DATAPROC_SPARK_CONNECT_IDLE_TTL_SECONDS": "89",
                "COLAB_NOTEBOOK_ID": "/embedded/projects/company.com%3Aproject1/locations/us-central1/repositories/test-notebook-id",
            },
        ).start()

        create_session_request = CreateSessionRequest()
        create_session_request.session.environment_config.execution_config.authentication_config.user_workload_authentication_type = (
            AuthenticationConfig.AuthenticationType.SERVICE_ACCOUNT
        )
        create_session_request.session.environment_config.execution_config.service_account = (
            "test-acc@example.com"
        )
        create_session_request.session.environment_config.execution_config.subnetwork_uri = (
            "test-subnet-from-env"
        )
        create_session_request.session.environment_config.execution_config.ttl = {
            "seconds": 12
        }
        create_session_request.session.environment_config.execution_config.idle_ttl = {
            "seconds": 89
        }
        create_session_request.session.labels["goog-colab-notebook-id"] = (
            "test-notebook-id"  # Expecting the basename
        )
        create_session_request.parent = (
            "projects/test-project/locations/test-region"
        )
        create_session_request.session.name = "projects/test-project/locations/test-region/sessions/sc-20240702-103952-abcdef"
        create_session_request.session.runtime_config.version = (
            self._default_runtime_version
        )
        create_session_request.session.spark_connect_session = (
            SparkConnectConfig()
        )
        create_session_request.session_id = "sc-20240702-103952-abcdef"
        create_session_request.session.labels["dataproc-session-client"] = (
            "unknown"
        )

        try:
            session = DataprocSparkSession.builder.getOrCreate()
            mock_session_controller_client_instance.create_session.assert_called_once_with(
                create_session_request
            )
        finally:
            mock_session_controller_client_instance.terminate_session.return_value = (
                mock.Mock()
            )
            self.stopSession(mock_session_controller_client_instance, session)
            terminate_session_request = TerminateSessionRequest()
            terminate_session_request.name = "projects/test-project/locations/test-region/sessions/sc-20240702-103952-abcdef"
            get_session_request = GetSessionRequest()
            get_session_request.name = "projects/test-project/locations/test-region/sessions/sc-20240702-103952-abcdef"
            mock_session_controller_client_instance.terminate_session.assert_called_once_with(
                terminate_session_request
            )
            mock_session_controller_client_instance.get_session.assert_called_once_with(
                get_session_request
            )

    @mock.patch("google.auth.default")
    @mock.patch("google.cloud.dataproc_v1.SessionControllerClient")
    @mock.patch("pyspark.sql.connect.client.SparkConnectClient.config")
    @mock.patch(
        "google.cloud.dataproc_spark_connect.DataprocSparkSession.Builder.generate_dataproc_session_id"
    )
    @mock.patch(
        "google.cloud.dataproc_spark_connect.session.is_s8s_session_active"
    )
    def test_create_session_with_session_template(
        self,
        mock_is_s8s_session_active,
        mock_dataproc_session_id,
        mock_client_config,
        mock_session_controller_client,
        mock_credentials,
    ):
        session = None
        mock_is_s8s_session_active.return_value = True
        mock_session_controller_client_instance = (
            mock_session_controller_client.return_value
        )
        mock_dataproc_session_id.return_value = "sc-20240702-103952-abcdef"
        mock_client_config.return_value = ConfigResult.fromProto(
            ConfigResponse()
        )
        cred = mock.MagicMock()
        cred.token = "token"
        mock_credentials.return_value = (cred, "")
        mock_operation = mock.Mock()
        session_response = Session()
        session_response.runtime_info.endpoints = {
            "Spark Connect Server": "sc://spark-connect-server.example.com:443"
        }
        session_response.uuid = "c002e4ef-fe5e-41a8-a157-160aa73e4f7f"
        mock_operation.result.side_effect = [session_response]
        mock_session_controller_client_instance.create_session.return_value = (
            mock_operation
        )

        create_session_request = CreateSessionRequest()
        create_session_request.parent = (
            "projects/test-project/locations/test-region"
        )
        create_session_request.session.name = "projects/test-project/locations/test-region/sessions/sc-20240702-103952-abcdef"
        create_session_request.session.runtime_config.version = (
            self._default_runtime_version
        )
        create_session_request.session.spark_connect_session = (
            SparkConnectConfig()
        )
        create_session_request.session_id = "sc-20240702-103952-abcdef"
        create_session_request.session.labels["dataproc-session-client"] = (
            "unknown"
        )
        create_session_request.session.session_template = "projects/test-project/locations/test-region/sessionTemplates/test_template"

        try:
            dataproc_config = Session()
            dataproc_config.session_template = "projects/test-project/locations/test-region/sessionTemplates/test_template"
            session = DataprocSparkSession.builder.dataprocSessionConfig(
                dataproc_config
            ).getOrCreate()
            mock_session_controller_client_instance.create_session.assert_called_once_with(
                create_session_request
            )
        finally:
            mock_session_controller_client_instance.terminate_session.return_value = (
                mock.Mock()
            )
            self.stopSession(mock_session_controller_client_instance, session)
            terminate_session_request = TerminateSessionRequest()
            terminate_session_request.name = "projects/test-project/locations/test-region/sessions/sc-20240702-103952-abcdef"
            get_session_request = GetSessionRequest()
            get_session_request.name = "projects/test-project/locations/test-region/sessions/sc-20240702-103952-abcdef"
            mock_session_controller_client_instance.terminate_session.assert_called_once_with(
                terminate_session_request
            )
            mock_session_controller_client_instance.get_session.assert_called_once_with(
                get_session_request
            )

    @mock.patch("google.auth.default")
    @mock.patch("google.cloud.dataproc_v1.SessionControllerClient")
    @mock.patch("pyspark.sql.connect.client.SparkConnectClient.config")
    @mock.patch(
        "google.cloud.dataproc_spark_connect.DataprocSparkSession.Builder.generate_dataproc_session_id"
    )
    @mock.patch(
        "google.cloud.dataproc_spark_connect.session.is_s8s_session_active"
    )
    def test_create_session_with_user_provided_dataproc_config_and_session_template(
        self,
        mock_is_s8s_session_active,
        mock_dataproc_session_id,
        mock_client_config,
        mock_session_controller_client,
        mock_credentials,
    ):
        session = None
        mock_is_s8s_session_active.return_value = True
        mock_session_controller_client_instance = (
            mock_session_controller_client.return_value
        )
        mock_dataproc_session_id.return_value = "sc-20240702-103952-abcdef"
        mock_client_config.return_value = ConfigResult.fromProto(
            ConfigResponse()
        )
        cred = mock.MagicMock()
        cred.token = "token"
        mock_credentials.return_value = (cred, "")
        mock_operation = mock.Mock()
        session_response = Session()
        session_response.runtime_info.endpoints = {
            "Spark Connect Server": "sc://spark-connect-server.example.com:443"
        }
        session_response.uuid = "c002e4ef-fe5e-41a8-a157-160aa73e4f7f"
        mock_operation.result.side_effect = [session_response]
        mock_session_controller_client_instance.create_session.return_value = (
            mock_operation
        )

        create_session_request = CreateSessionRequest()
        create_session_request.parent = (
            "projects/test-project/locations/test-region"
        )
        create_session_request.session.environment_config.execution_config.ttl = {
            "seconds": 10
        }
        create_session_request.session.name = "projects/test-project/locations/test-region/sessions/sc-20240702-103952-abcdef"
        create_session_request.session.runtime_config.version = (
            self._default_runtime_version
        )
        create_session_request.session.spark_connect_session = (
            SparkConnectConfig()
        )
        create_session_request.session.session_template = "projects/test-project/locations/test-region/sessionTemplates/test_template"
        create_session_request.session_id = "sc-20240702-103952-abcdef"
        create_session_request.session.labels["dataproc-session-client"] = (
            "unknown"
        )

        try:
            dataproc_config = Session()
            dataproc_config.environment_config.execution_config.ttl = {
                "seconds": 10
            }
            dataproc_config.session_template = "projects/test-project/locations/test-region/sessionTemplates/test_template"
            session = DataprocSparkSession.builder.dataprocSessionConfig(
                dataproc_config
            ).getOrCreate()
            mock_session_controller_client_instance.create_session.assert_called_once_with(
                create_session_request
            )
        finally:
            mock_session_controller_client_instance.terminate_session.return_value = (
                mock.Mock()
            )
            self.stopSession(mock_session_controller_client_instance, session)
            terminate_session_request = TerminateSessionRequest()
            terminate_session_request.name = "projects/test-project/locations/test-region/sessions/sc-20240702-103952-abcdef"
            get_session_request = GetSessionRequest()
            get_session_request.name = "projects/test-project/locations/test-region/sessions/sc-20240702-103952-abcdef"
            mock_session_controller_client_instance.terminate_session.assert_called_once_with(
                terminate_session_request
            )
            mock_session_controller_client_instance.get_session.assert_called_once_with(
                get_session_request
            )

    @mock.patch("google.auth.default")
    @mock.patch("google.cloud.dataproc_v1.SessionControllerClient")
    @mock.patch(
        "google.cloud.dataproc_spark_connect.DataprocSparkSession.Builder.generate_dataproc_session_id"
    )
    def test_create_spark_session_with_create_session_failed(
        self,
        mock_dataproc_session_id,
        mock_session_controller_client,
        mock_credentials,
    ):
        mock_dataproc_session_id.return_value = "sc-20240702-103952-abcdef"
        mock_session_controller_client_instance = (
            mock_session_controller_client.return_value
        )
        mock_operation = mock.Mock()
        mock_operation.result.side_effect = Exception(
            "Testing create session failure"
        )
        mock_session_controller_client_instance.create_session.return_value = (
            mock_operation
        )
        cred = mock.MagicMock()
        cred.token = "token"
        mock_credentials.return_value = (cred, "")
        with self.assertRaises(RuntimeError) as e:
            DataprocSparkSession.builder.dataprocSessionConfig(
                Session()
            ).getOrCreate()
        self.assertEqual(
            "Error while creating Dataproc Session", e.exception.args[0]
        )

    @mock.patch("google.auth.default")
    @mock.patch("google.cloud.dataproc_v1.SessionControllerClient")
    def test_create_spark_session_with_invalid_argument(
        self,
        mock_session_controller_client,
        mock_credentials,
    ):
        mock_session_controller_client_instance = (
            mock_session_controller_client.return_value
        )
        mock_operation = mock.Mock()
        mock_operation.result.side_effect = InvalidArgument(
            "Network does not have permissions"
        )
        mock_session_controller_client_instance.create_session.return_value = (
            mock_operation
        )
        cred = mock.MagicMock()
        cred.token = "token"
        mock_credentials.return_value = (cred, "")
        with self.assertRaises(DataprocSparkConnectException) as e:
            DataprocSparkSession.builder.dataprocSessionConfig(
                Session()
            ).getOrCreate()
            self.assertEqual(
                e.exception.error_message,
                "Error while creating Dataproc Session: "
                "400 Network does not have permissions",
            )

    @mock.patch("google.auth.default")
    @mock.patch("google.cloud.dataproc_v1.SessionControllerClient")
    @mock.patch(
        "google.cloud.dataproc_spark_connect.DataprocSparkSession.Builder.generate_dataproc_session_id"
    )
    @mock.patch(
        "google.cloud.dataproc_spark_connect.session.is_s8s_session_active"
    )
    def test_spark_session_with_inactive_s8s_session(
        self,
        mock_is_s8s_session_active,
        mock_dataproc_session_id,
        mock_session_controller_client,
        mock_credentials,
    ):
        session = None
        mock_is_s8s_session_active.return_value = False
        mock_session_controller_client_instance = (
            mock_session_controller_client.return_value
        )

        mock_dataproc_session_id.return_value = "sc-20240702-103952-abcdef"

        cred = mock.MagicMock()
        cred.token = "token"
        mock_credentials.return_value = (cred, "")
        mock_operation = mock.Mock()
        session_response = Session()
        session_response.runtime_info.endpoints = {
            "Spark Connect Server": "sc://spark-connect-server.example.com:443"
        }
        session_response.uuid = "c002e4ef-fe5e-41a8-a157-160aa73e4f7f"
        mock_operation.result.side_effect = [session_response]
        mock_session_controller_client_instance.create_session.return_value = (
            mock_operation
        )
        with self.assertRaises(RuntimeError) as e:
            session = DataprocSparkSession.builder.getOrCreate()
            session.createDataFrame([(1, "Sarah"), (2, "Maria")]).toDF(
                "id", "name"
            ).show()
            self.assertEqual(
                e.exception.args[0],
                "Session not active. Please create a new session ",
            )
        self.stopSession(mock_session_controller_client_instance, session)

    @mock.patch("pyspark.sql.connect.client.SparkConnectClient.config")
    @mock.patch("google.auth.default")
    @mock.patch("google.cloud.dataproc_v1.SessionControllerClient")
    @mock.patch(
        "google.cloud.dataproc_spark_connect.session.is_s8s_session_active"
    )
    def test_stop_spark_session_with_terminated_s8s_session(
        self,
        mock_is_s8s_session_active,
        mock_session_controller_client,
        mock_credentials,
        mock_client_config,
    ):
        session = None
        mock_is_s8s_session_active.return_value = True
        mock_session_controller_client_instance = (
            mock_session_controller_client.return_value
        )
        try:
            mock_operation = mock.Mock()
            session_response = Session()
            session_response.runtime_info.endpoints = {
                "Spark Connect Server": "sc://spark-connect-server.example.com:443"
            }
            session_response.uuid = "c002e4ef-fe5e-41a8-a157-160aa73e4f7f"
            mock_operation.result.side_effect = [session_response]
            mock_session_controller_client_instance.create_session.return_value = (
                mock_operation
            )
            cred = mock.MagicMock()
            cred.token = "token"
            mock_credentials.return_value = (cred, "")
            mock_client_config.return_value = ConfigResult.fromProto(
                ConfigResponse()
            )
            session = DataprocSparkSession.builder.getOrCreate()

        finally:
            mock_session_controller_client_instance.terminate_session.side_effect = FailedPrecondition(
                "Already terminated"
            )
            if session is not None:
                session.stop()
            self.assertIsNone(DataprocSparkSession._active_s8s_session_uuid)

    @mock.patch("pyspark.sql.connect.client.SparkConnectClient.config")
    @mock.patch("google.auth.default")
    @mock.patch("google.cloud.dataproc_v1.SessionControllerClient")
    @mock.patch(
        "google.cloud.dataproc_spark_connect.session.is_s8s_session_active"
    )
    def test_stop_spark_session_with_creating_s8s_session(
        self,
        mock_is_s8s_session_active,
        mock_session_controller_client,
        mock_credentials,
        mock_client_config,
    ):
        session = None
        mock_is_s8s_session_active.return_value = True
        mock_session_controller_client_instance = (
            mock_session_controller_client.return_value
        )
        try:
            mock_operation = mock.Mock()
            session_response = Session()
            session_response.runtime_info.endpoints = {
                "Spark Connect Server": "sc://spark-connect-server.example.com:443"
            }
            session_response.uuid = "c002e4ef-fe5e-41a8-a157-160aa73e4f7f"
            mock_operation.result.side_effect = [session_response]
            mock_session_controller_client_instance.create_session.return_value = (
                mock_operation
            )
            cred = mock.MagicMock()
            cred.token = "token"
            mock_credentials.return_value = (cred, "")
            mock_client_config.return_value = ConfigResult.fromProto(
                ConfigResponse()
            )
            session = DataprocSparkSession.builder.getOrCreate()

        finally:
            mock_session_controller_client_instance.terminate_session.side_effect = Aborted(
                "still being created"
            )
            if session is not None:
                session.stop()
            self.assertIsNone(DataprocSparkSession._active_s8s_session_uuid)

    @mock.patch("pyspark.sql.connect.client.SparkConnectClient.config")
    @mock.patch("google.auth.default")
    @mock.patch("google.cloud.dataproc_v1.SessionControllerClient")
    @mock.patch(
        "google.cloud.dataproc_spark_connect.session.is_s8s_session_active"
    )
    def test_stop_spark_session_with_deleted_s8s_session(
        self,
        mock_is_s8s_session_active,
        mock_session_controller_client,
        mock_credentials,
        mock_client_config,
    ):
        session = None
        mock_is_s8s_session_active.return_value = True
        mock_session_controller_client_instance = (
            mock_session_controller_client.return_value
        )
        try:
            mock_operation = mock.Mock()
            session_response = Session()
            session_response.runtime_info.endpoints = {
                "Spark Connect Server": "sc://spark-connect-server.example.com:443"
            }
            session_response.uuid = "c002e4ef-fe5e-41a8-a157-160aa73e4f7f"
            mock_operation.result.side_effect = [session_response]
            mock_session_controller_client_instance.create_session.return_value = (
                mock_operation
            )
            cred = mock.MagicMock()
            cred.token = "token"
            mock_credentials.return_value = (cred, "")
            mock_client_config.return_value = ConfigResult.fromProto(
                ConfigResponse()
            )
            session = DataprocSparkSession.builder.getOrCreate()

        finally:
            mock_session_controller_client_instance.terminate_session.side_effect = NotFound(
                "Already deleted"
            )
            if session is not None:
                session.stop()
            self.assertIsNone(DataprocSparkSession._active_s8s_session_uuid)

    @mock.patch("pyspark.sql.connect.client.SparkConnectClient.config")
    @mock.patch("google.auth.default")
    @mock.patch("google.cloud.dataproc_v1.SessionControllerClient")
    @mock.patch(
        "google.cloud.dataproc_spark_connect.DataprocSparkSession.Builder.generate_dataproc_session_id"
    )
    @mock.patch(
        "google.cloud.dataproc_spark_connect.session.is_s8s_session_active"
    )
    def test_stop_spark_session_wait_for_terminating_state(
        self,
        mock_is_s8s_session_active,
        mock_dataproc_session_id,
        mock_session_controller_client,
        mock_credentials,
        mock_client_config,
    ):
        session = None
        mock_is_s8s_session_active.return_value = True
        mock_dataproc_session_id.return_value = "sc-20240702-103952-abcdef"
        mock_session_controller_client_instance = (
            mock_session_controller_client.return_value
        )
        try:
            mock_operation = mock.Mock()
            session_response = Session()
            session_response.runtime_info.endpoints = {
                "Spark Connect Server": "sc://spark-connect-server.example.com:443"
            }
            session_response.uuid = "c002e4ef-fe5e-41a8-a157-160aa73e4f7f"
            mock_operation.result.side_effect = [session_response]
            mock_session_controller_client_instance.create_session.return_value = (
                mock_operation
            )
            cred = mock.MagicMock()
            cred.token = "token"
            mock_credentials.return_value = (cred, "")
            mock_client_config.return_value = ConfigResult.fromProto(
                ConfigResponse()
            )
            session = DataprocSparkSession.builder.getOrCreate()

        finally:
            mock_session_controller_client_instance.terminate_session.return_value = (
                mock.Mock()
            )
            self.stopSession(mock_session_controller_client_instance, session)
            terminate_session_request = TerminateSessionRequest()
            terminate_session_request.name = "projects/test-project/locations/test-region/sessions/sc-20240702-103952-abcdef"
            get_session_request = GetSessionRequest()
            get_session_request.name = "projects/test-project/locations/test-region/sessions/sc-20240702-103952-abcdef"
            mock_session_controller_client_instance.terminate_session.assert_called_once_with(
                terminate_session_request
            )
            mock_session_controller_client_instance.get_session.assert_called_once_with(
                get_session_request
            )

    @mock.patch("google.auth.default")
    @mock.patch("google.cloud.dataproc_v1.SessionControllerClient")
    @mock.patch("pyspark.sql.connect.client.SparkConnectClient.config")
    @mock.patch(
        "google.cloud.dataproc_spark_connect.DataprocSparkSession.Builder.generate_dataproc_session_id"
    )
    @mock.patch(
        "google.cloud.dataproc_spark_connect.session.is_s8s_session_active"
    )
    @mock.patch(
        "google.cloud.dataproc_spark_connect.session.logger"
    )  # Mock the logger
    def test_create_session_with_default_datasource_env_var(
        self,
        mock_logger,  # Add mock logger parameter
        mock_is_s8s_session_active,
        mock_dataproc_session_id,
        mock_client_config,
        mock_session_controller_client,
        mock_credentials,
    ):
        session = None
        mock_is_s8s_session_active.return_value = True
        mock_session_controller_client_instance = (
            mock_session_controller_client.return_value
        )
        mock_dataproc_session_id.return_value = (
            "c002e4ef-fe5e-41a8-a157-160aa73e4f7f"  # Use a valid UUID
        )
        mock_client_config.return_value = ConfigResult.fromProto(
            ConfigResponse()
        )
        cred = mock.MagicMock()
        cred.token = "token"
        mock_credentials.return_value = (cred, "")
        mock_operation = mock.Mock()
        session_response = Session()
        session_response.runtime_info.endpoints = {
            "Spark Connect Server": "sc://spark-connect-server.example.com:443"
        }
        session_response.uuid = (
            "c002e4ef-fe5e-41a8-a157-160aa73e4f7f"  # Use a valid UUID
        )
        mock_operation.result.side_effect = [
            session_response,
            session_response,
            session_response,
            session_response,
            session_response,
            session_response,
        ]  # Provide a response for each getOrCreate call
        mock_session_controller_client_instance.create_session.return_value = (
            mock_operation
        )

        # Scenario 1: DATAPROC_SPARK_CONNECT_DEFAULT_DATASOURCE is not set
        with mock.patch.dict(os.environ, {}, clear=True):
            os.environ["GOOGLE_CLOUD_PROJECT"] = "test-project"
            os.environ["GOOGLE_CLOUD_REGION"] = "test-region"
            session = DataprocSparkSession.builder.getOrCreate()
            create_session_request = mock_session_controller_client_instance.create_session.call_args[
                0
            ][
                0
            ]
            self.assertNotIn(
                "spark.datasource.bigquery.writeMethod",
                create_session_request.session.runtime_config.properties,
            )
            mock_logger.warning.assert_not_called()
            self.stopSession(mock_session_controller_client_instance, session)
            mock_session_controller_client_instance.create_session.reset_mock()
            mock_logger.warning.reset_mock()

        # Scenario 2: DATAPROC_SPARK_CONNECT_DEFAULT_DATASOURCE is set to "bigquery"
        with mock.patch.dict(
            os.environ,
            {"DATAPROC_SPARK_CONNECT_DEFAULT_DATASOURCE": "bigquery"},
            clear=True,
        ):
            os.environ["GOOGLE_CLOUD_PROJECT"] = "test-project"
            os.environ["GOOGLE_CLOUD_REGION"] = "test-region"
            session = DataprocSparkSession.builder.getOrCreate()
            create_session_request = mock_session_controller_client_instance.create_session.call_args[
                0
            ][
                0
            ]
            # BigQuery properties should be set
            self.assertEqual(
                create_session_request.session.runtime_config.properties.get(
                    "spark.datasource.bigquery.writeMethod"
                ),
                "direct",
            )
            self.assertEqual(
                create_session_request.session.runtime_config.properties.get(
                    "spark.datasource.bigquery.viewsEnabled"
                ),
                "true",
            )
            self.assertEqual(
                create_session_request.session.runtime_config.properties.get(
                    "spark.sql.sources.default"
                ),
                "bigquery",
            )
            self.assertEqual(
                create_session_request.session.runtime_config.properties.get(
                    "spark.sql.catalog.spark_catalog"
                ),
                "com.google.cloud.spark.bigquery.BigQuerySparkSessionCatalog",
            )
            mock_logger.warning.assert_not_called()
            self.stopSession(mock_session_controller_client_instance, session)
            mock_session_controller_client_instance.create_session.reset_mock()
            mock_logger.warning.reset_mock()

        # Scenario 3: DATAPROC_SPARK_CONNECT_DEFAULT_DATASOURCE is set to an invalid value
        with mock.patch.dict(
            os.environ,
            {"DATAPROC_SPARK_CONNECT_DEFAULT_DATASOURCE": "invalid_datasource"},
            clear=True,
        ):
            os.environ["GOOGLE_CLOUD_PROJECT"] = "test-project"
            os.environ["GOOGLE_CLOUD_REGION"] = "test-region"
            session = DataprocSparkSession.builder.getOrCreate()
            create_session_request = mock_session_controller_client_instance.create_session.call_args[
                0
            ][
                0
            ]
            self.assertNotIn(
                "spark.datasource.bigquery.writeMethod",
                create_session_request.session.runtime_config.properties,
            )
            mock_logger.warning.assert_called_once_with(
                "DATAPROC_SPARK_CONNECT_DEFAULT_DATASOURCE is set to an invalid value: invalid_datasource. Supported value is 'bigquery'."
            )
            self.stopSession(mock_session_controller_client_instance, session)
            mock_session_controller_client_instance.create_session.reset_mock()
            mock_logger.warning.reset_mock()

        # Scenario 4: DATAPROC_SPARK_CONNECT_DEFAULT_DATASOURCE is set to "bigquery" with pre-existing properties
        with mock.patch.dict(
            os.environ,
            {"DATAPROC_SPARK_CONNECT_DEFAULT_DATASOURCE": "bigquery"},
            clear=True,
        ):
            os.environ["GOOGLE_CLOUD_PROJECT"] = "test-project"
            os.environ["GOOGLE_CLOUD_REGION"] = "test-region"
            dataproc_config = Session()
            dataproc_config.runtime_config.version = "3.0"
            dataproc_config.runtime_config.properties = {
                "spark.datasource.bigquery.writeMethod": "override_method",
                "spark.some.other.property": "some_value",
            }
            session = DataprocSparkSession.builder.dataprocSessionConfig(
                dataproc_config
            ).getOrCreate()
            create_session_request = mock_session_controller_client_instance.create_session.call_args[
                0
            ][
                0
            ]
            # The BigQuery default properties should be set,
            # but pre-existing properties should override defaults.
            self.assertEqual(
                create_session_request.session.runtime_config.properties.get(
                    "spark.datasource.bigquery.writeMethod"
                ),
                "override_method",
            )  # Pre-existing property remains
            self.assertEqual(
                create_session_request.session.runtime_config.properties.get(
                    "spark.datasource.bigquery.viewsEnabled"
                ),
                "true",
            )  # Default should still be set
            self.assertEqual(
                create_session_request.session.runtime_config.properties.get(
                    "spark.sql.sources.default"
                ),
                "bigquery",
            )  # Default should still be set
            self.assertEqual(
                create_session_request.session.runtime_config.properties.get(
                    "spark.sql.catalog.spark_catalog"
                ),
                "com.google.cloud.spark.bigquery.BigQuerySparkSessionCatalog",
            )  # Default should still be set
            self.assertEqual(
                create_session_request.session.runtime_config.properties.get(
                    "spark.some.other.property"
                ),
                "some_value",
            )  # Existing property should remain
            mock_logger.warning.assert_not_called()
            self.stopSession(mock_session_controller_client_instance, session)
            mock_session_controller_client_instance.create_session.reset_mock()
            mock_logger.warning.reset_mock()

    @mock.patch.dict("sys.modules", {"google.cloud.aiplatform": None})
    @mock.patch(
        "IPython.core.interactiveshell.InteractiveShell.initialized",
        return_value=True,
    )
    @mock.patch("google.cloud.dataproc_spark_connect.session.logger")
    def test_display_button_with_aiplatform_not_installed(
        self, mock_logger, _mock_ipy
    ):
        mock.patch.dict(
            os.environ,
            {
                "VERTEX_PRODUCT": "COLAB_ENTERPRISE",
            },
        ).start()
        DataprocSparkSession.builder._display_view_session_details_button(
            "test_session"
        )
        mock_logger.debug.assert_called_once_with(
            "Import error: No module named 'google.cloud.aiplatform.utils'; 'google.cloud.aiplatform' is not a package"
        )

    @mock.patch.dict(
        "sys.modules",
        {
            "google.cloud.aiplatform.utils": mock.MagicMock(
                _ipython_utils=mock.MagicMock()
            ),
        },
    )
    @mock.patch(
        "IPython.core.interactiveshell.InteractiveShell.initialized",
        return_value=True,
    )
    def test_display_button_with_aiplatform_installed_ipython_interactive(
        self, _mock_ipy
    ):
        mock.patch.dict(
            os.environ,
            {
                "VERTEX_PRODUCT": "COLAB_ENTERPRISE",
            },
        ).start()
        mock_ipython_utils = mock.sys.modules[
            "google.cloud.aiplatform.utils"
        ]._ipython_utils
        test_session_url = "https://console.cloud.google.com/dataproc/interactive/sessions/test_session/locations/test-region?project=test-project"

        mock_display_link = mock_ipython_utils.display_link
        DataprocSparkSession.builder._display_view_session_details_button(
            "test_session"
        )
        mock_display_link.assert_called_once_with(
            "View Session Details", test_session_url, "dashboard"
        )

    @mock.patch.dict(
        "sys.modules",
        {
            "google.cloud.aiplatform.utils": mock.MagicMock(
                _ipython_utils=mock.MagicMock()
            ),
        },
    )
    @mock.patch(
        "IPython.core.interactiveshell.InteractiveShell.initialized",
        return_value=False,
    )
    def test_display_button_with_aiplatform_installed_ipython_non_interactive(
        self, _mock_ipy
    ):
        mock.patch.dict(
            os.environ,
            {
                "VERTEX_PRODUCT": "COLAB_ENTERPRISE",
            },
        ).start()
        mock_ipython_utils = mock.sys.modules[
            "google.cloud.aiplatform.utils"
        ]._ipython_utils

        mock_display_link = mock_ipython_utils.display_link
        DataprocSparkSession.builder._display_view_session_details_button(
            "test_session"
        )
        mock_display_link.assert_not_called()

    def test_is_valid_label_value(self):
        # Valid label values
        self.assertTrue(_is_valid_label_value("valid-label-123"))
        self.assertTrue(_is_valid_label_value("123"))
        self.assertTrue(_is_valid_label_value("a"))
        self.assertTrue(_is_valid_label_value("test-notebook-id"))
        self.assertTrue(_is_valid_label_value("a1b2c3"))
        self.assertTrue(_is_valid_label_value("valid123"))
        self.assertTrue(_is_valid_label_value("123valid"))

        # Invalid label values
        self.assertFalse(_is_valid_label_value(""))  # Empty string
        self.assertFalse(
            _is_valid_label_value("Invalid-Capital")
        )  # Capital letters
        self.assertFalse(_is_valid_label_value("-invalid"))  # Starts with dash
        self.assertFalse(_is_valid_label_value("invalid-"))  # Ends with dash
        self.assertFalse(
            _is_valid_label_value("invalid_underscore")
        )  # Contains underscore
        self.assertFalse(_is_valid_label_value("invalid.dot"))  # Contains dot
        self.assertFalse(
            _is_valid_label_value("invalid spaces")
        )  # Contains spaces
        self.assertFalse(
            _is_valid_label_value("invalid@symbol")
        )  # Contains special char
        self.assertFalse(_is_valid_label_value("UPPERCASE"))  # All uppercase
        self.assertFalse(_is_valid_label_value("-"))  # Just a dash

        # Valid label value at maximum length (63 characters)
        max_length_valid = "a" + "b" * 61 + "c"  # 63 characters: a + 61 b's + c
        self.assertEqual(len(max_length_valid), 63)
        self.assertTrue(_is_valid_label_value(max_length_valid))

        # Invalid label value - too long (64 characters)
        too_long_invalid = "a" + "b" * 62 + "c"  # 64 characters: a + 62 b's + c
        self.assertEqual(len(too_long_invalid), 64)
        self.assertFalse(_is_valid_label_value(too_long_invalid))

    @mock.patch("google.auth.default")
    @mock.patch("google.cloud.dataproc_v1.SessionControllerClient")
    @mock.patch(
        "google.cloud.dataproc_spark_connect.DataprocSparkSession.Builder.generate_dataproc_session_id"
    )
    @mock.patch(
        "google.cloud.dataproc_spark_connect.session.is_s8s_session_active"
    )
    @mock.patch("google.cloud.dataproc_spark_connect.session.logger")
    def test_create_session_with_invalid_notebook_id(
        self,
        mock_logger,
        mock_is_s8s_session_active,
        mock_dataproc_session_id,
        mock_session_controller_client,
        mock_credentials,
    ):
        session = None
        mock_is_s8s_session_active.return_value = True
        mock_session_controller_client_instance = (
            mock_session_controller_client.return_value
        )
        mock_dataproc_session_id.return_value = "sc-20240702-103952-abcdef"
        cred = mock.MagicMock()
        cred.token = "token"
        mock_credentials.return_value = (cred, "")
        mock_operation = mock.Mock()
        session_response = Session()
        session_response.runtime_info.endpoints = {
            "Spark Connect Server": "sc://spark-connect-server.example.com:443"
        }
        session_response.uuid = "c002e4ef-fe5e-41a8-a157-160aa73e4f7f"
        mock_operation.result.side_effect = [session_response]
        mock_session_controller_client_instance.create_session.return_value = (
            mock_operation
        )

        # Test with invalid notebook ID (contains uppercase and underscores)
        mock.patch.dict(
            os.environ,
            {
                "COLAB_NOTEBOOK_ID": "/path/to/Invalid_Notebook-ID_With.Special@Chars",
            },
        ).start()

        create_session_request = CreateSessionRequest()
        create_session_request.parent = (
            "projects/test-project/locations/test-region"
        )
        create_session_request.session.name = "projects/test-project/locations/test-region/sessions/sc-20240702-103952-abcdef"
        create_session_request.session.runtime_config.version = (
            self._default_runtime_version
        )
        create_session_request.session.spark_connect_session = (
            SparkConnectConfig()
        )
        create_session_request.session_id = "sc-20240702-103952-abcdef"
        create_session_request.session.labels["dataproc-session-client"] = (
            "unknown"
        )
        # Note: No notebook label should be set due to invalid format

        try:
            session = DataprocSparkSession.builder.getOrCreate()
            mock_session_controller_client_instance.create_session.assert_called_once_with(
                create_session_request
            )
            # Verify warning was logged
            mock_logger.warning.assert_called_once()
            warning_call_args = mock_logger.warning.call_args[0][0]
            self.assertIn(
                "Warning while processing notebook ID:", warning_call_args
            )
            self.assertIn(
                "Invalid_Notebook-ID_With.Special@Chars", warning_call_args
            )
            self.assertIn(
                "not compliant with label value format", warning_call_args
            )
            self.assertIn(
                "Only lowercase letters, numbers, and dashes are allowed",
                warning_call_args,
            )
            self.assertIn("Maximum length is 63 characters", warning_call_args)
            self.assertIn("Ignoring notebook ID label", warning_call_args)

        finally:
            mock_session_controller_client_instance.terminate_session.return_value = (
                mock.Mock()
            )
            self.stopSession(mock_session_controller_client_instance, session)

    @mock.patch("google.auth.default")
    @mock.patch("google.cloud.dataproc_v1.SessionControllerClient")
    @mock.patch(
        "google.cloud.dataproc_spark_connect.DataprocSparkSession.Builder.generate_dataproc_session_id"
    )
    @mock.patch(
        "google.cloud.dataproc_spark_connect.session.is_s8s_session_active"
    )
    @mock.patch("google.cloud.dataproc_spark_connect.session.logger")
    def test_create_session_with_valid_notebook_id(
        self,
        mock_logger,
        mock_is_s8s_session_active,
        mock_dataproc_session_id,
        mock_session_controller_client,
        mock_credentials,
    ):
        session = None
        mock_is_s8s_session_active.return_value = True
        mock_session_controller_client_instance = (
            mock_session_controller_client.return_value
        )
        mock_dataproc_session_id.return_value = "sc-20240702-103952-abcdef"
        cred = mock.MagicMock()
        cred.token = "token"
        mock_credentials.return_value = (cred, "")
        mock_operation = mock.Mock()
        session_response = Session()
        session_response.runtime_info.endpoints = {
            "Spark Connect Server": "sc://spark-connect-server.example.com:443"
        }
        session_response.uuid = "c002e4ef-fe5e-41a8-a157-160aa73e4f7f"
        mock_operation.result.side_effect = [session_response]
        mock_session_controller_client_instance.create_session.return_value = (
            mock_operation
        )

        # Test with valid notebook ID (lowercase, numbers, dashes only)
        mock.patch.dict(
            os.environ,
            {
                "COLAB_NOTEBOOK_ID": "/path/to/valid-notebook-123",
            },
        ).start()

        create_session_request = CreateSessionRequest()
        create_session_request.parent = (
            "projects/test-project/locations/test-region"
        )
        create_session_request.session.name = "projects/test-project/locations/test-region/sessions/sc-20240702-103952-abcdef"
        create_session_request.session.runtime_config.version = (
            self._default_runtime_version
        )
        create_session_request.session.spark_connect_session = (
            SparkConnectConfig()
        )
        create_session_request.session_id = "sc-20240702-103952-abcdef"
        create_session_request.session.labels["dataproc-session-client"] = (
            "unknown"
        )
        # Valid notebook label should be set
        create_session_request.session.labels["goog-colab-notebook-id"] = (
            "valid-notebook-123"
        )

        try:
            session = DataprocSparkSession.builder.getOrCreate()
            mock_session_controller_client_instance.create_session.assert_called_once_with(
                create_session_request
            )
            # Verify no warning was logged
            mock_logger.warning.assert_not_called()

        finally:
            mock_session_controller_client_instance.terminate_session.return_value = (
                mock.Mock()
            )
            self.stopSession(mock_session_controller_client_instance, session)


class DataprocSparkConnectClientTest(unittest.TestCase):

    @staticmethod
    def stopSession(mock_session_controller_client_instance, session):
        session_response = Session()
        session_response.state = Session.State.TERMINATING
        mock_session_controller_client_instance.get_session.return_value = (
            session_response
        )
        if session is not None:
            session.stop()

    @staticmethod
    def _setup_session_creation_mocks(
        mock_is_s8s_session_active,
        mock_dataproc_session_id,
        mock_client_config,
        mock_session_controller_client,
        mock_credentials,
        session_id="sc-20240702-103952-abcdef",
        session_uuid="c002e4ef-fe5e-41a8-a157-160aa73e4f7f",
    ):
        """Helper method to set up common mocks for session creation tests."""
        mock_is_s8s_session_active.return_value = True
        mock_session_controller_client_instance = (
            mock_session_controller_client.return_value
        )
        mock_dataproc_session_id.return_value = session_id
        mock_client_config.return_value = ConfigResult.fromProto(
            ConfigResponse()
        )

        cred = mock.MagicMock()
        cred.token = "token"
        mock_credentials.return_value = (cred, "")

        mock_operation = mock.Mock()
        session_response = Session()
        session_response.runtime_info.endpoints = {
            "Spark Connect Server": "sc://spark-connect-server.example.com:443"
        }
        session_response.uuid = session_uuid
        mock_operation.result.side_effect = [session_response]
        mock_session_controller_client_instance.create_session.return_value = (
            mock_operation
        )

        return mock_session_controller_client_instance

    @mock.patch("google.auth.default")
    @mock.patch("google.cloud.dataproc_v1.SessionControllerClient")
    @mock.patch("pyspark.sql.connect.client.SparkConnectClient.config")
    @mock.patch(
        "google.cloud.dataproc_spark_connect.DataprocSparkSession.Builder.generate_dataproc_session_id"
    )
    @mock.patch(
        "google.cloud.dataproc_spark_connect.session.is_s8s_session_active"
    )
    @mock.patch("uuid.uuid4")
    @mock.patch(
        "pyspark.sql.connect.client.SparkConnectClient._execute_plan_request_with_metadata"
    )
    def test_execute_plan_request_default_behaviour(
        self,
        mock_super_execute_plan_request,
        mock_uuid4,
        mock_is_s8s_session_active,
        mock_dataproc_session_id,
        mock_client_config,
        mock_session_controller_client,
        mock_credentials,
    ):
        test_uuid = "f728f1b4-00a7-4e6e-8365-d12b4a7d42ab"
        test_execute_plan_request: ExecutePlanRequest = ExecutePlanRequest(
            session_id="mock-session_id-from-super",
            client_type="mock-client_type-from-super",
            tags=["mock-tag-from-super"],
            user_context=UserContext(user_id="mock-user-from-super"),
            operation_id=None,
        )

        session = None
        mock_super_execute_plan_request.return_value = deepcopy(
            test_execute_plan_request
        )
        mock_uuid4.return_value = test_uuid
        mock_is_s8s_session_active.return_value = True
        mock_session_controller_client_instance = (
            mock_session_controller_client.return_value
        )

        mock_dataproc_session_id.return_value = "sc-20240702-103952-abcdef"
        mock_client_config.return_value = ConfigResult.fromProto(
            ConfigResponse()
        )
        cred = mock.MagicMock()
        cred.token = "token"
        mock_credentials.return_value = (cred, "")
        mock_operation = mock.Mock()
        session_response = Session()
        session_response.runtime_info.endpoints = {
            "Spark Connect Server": "sc://spark-connect-server.example.com:443"
        }
        session_response.uuid = "c002e4ef-fe5e-41a8-a157-160aa73e4f7f"
        mock_operation.result.side_effect = [session_response]
        mock_session_controller_client_instance.create_session.return_value = (
            mock_operation
        )

        try:
            session = DataprocSparkSession.builder.getOrCreate()
            client = session.client

            result_request = client._execute_plan_request_with_metadata()

            self.assertEqual(result_request.operation_id, test_uuid)

            mock_super_execute_plan_request.assert_called_once()
            mock_uuid4.assert_called_once()

            self.assertEqual(
                result_request.session_id, test_execute_plan_request.session_id
            )
            self.assertEqual(
                result_request.client_type,
                test_execute_plan_request.client_type,
            )
            self.assertEqual(
                result_request.tags, test_execute_plan_request.tags
            )
            self.assertEqual(
                result_request.user_context.user_id,
                test_execute_plan_request.user_context.user_id,
            )

        finally:
            mock_session_controller_client_instance.terminate_session.return_value = (
                mock.Mock()
            )
            self.stopSession(mock_session_controller_client_instance, session)

    @mock.patch("google.auth.default")
    @mock.patch("google.cloud.dataproc_v1.SessionControllerClient")
    @mock.patch("pyspark.sql.connect.client.SparkConnectClient.config")
    @mock.patch(
        "google.cloud.dataproc_spark_connect.DataprocSparkSession.Builder.generate_dataproc_session_id"
    )
    @mock.patch(
        "google.cloud.dataproc_spark_connect.session.is_s8s_session_active"
    )
    @mock.patch("uuid.uuid4")
    @mock.patch(
        "pyspark.sql.connect.client.SparkConnectClient._execute_plan_request_with_metadata"
    )
    def test_execute_plan_request_with_operation_id_provided(
        self,
        mock_super_execute_plan_request,
        mock_uuid4,
        mock_is_s8s_session_active,
        mock_dataproc_session_id,
        mock_client_config,
        mock_session_controller_client,
        mock_credentials,
    ):
        test_uuid = "f728f1b4-00a7-4e6e-8365-d12b4a7d42ab"
        provided_uuid = "d27f4fc9-f627-4b72-b20a-aebb2481df74"
        test_execute_plan_request: ExecutePlanRequest = ExecutePlanRequest(
            session_id="mock-session_id-from-super",
            client_type="mock-client_type-from-super",
            tags=["mock-tag-from-super"],
            user_context=UserContext(user_id="mock-user-from-super"),
            operation_id=provided_uuid,
        )

        session = None
        mock_super_execute_plan_request.return_value = deepcopy(
            test_execute_plan_request
        )
        mock_uuid4.return_value = test_uuid
        mock_is_s8s_session_active.return_value = True
        mock_session_controller_client_instance = (
            mock_session_controller_client.return_value
        )

        mock_dataproc_session_id.return_value = "sc-20240702-103952-abcdef"
        mock_client_config.return_value = ConfigResult.fromProto(
            ConfigResponse()
        )
        cred = mock.MagicMock()
        cred.token = "token"
        mock_credentials.return_value = (cred, "")
        mock_operation = mock.Mock()
        session_response = Session()
        session_response.runtime_info.endpoints = {
            "Spark Connect Server": "sc://spark-connect-server.example.com:443"
        }
        session_response.uuid = "c002e4ef-fe5e-41a8-a157-160aa73e4f7f"
        mock_operation.result.side_effect = [session_response]
        mock_session_controller_client_instance.create_session.return_value = (
            mock_operation
        )

        try:
            session = DataprocSparkSession.builder.getOrCreate()
            client = session.client

            result_request = client._execute_plan_request_with_metadata()

            mock_super_execute_plan_request.assert_called_once()
            mock_uuid4.assert_not_called()

            self.assertEqual(result_request.operation_id, provided_uuid)
            self.assertEqual(
                result_request.session_id, test_execute_plan_request.session_id
            )
            self.assertEqual(
                result_request.client_type,
                test_execute_plan_request.client_type,
            )
            self.assertEqual(
                result_request.tags, test_execute_plan_request.tags
            )
            self.assertEqual(
                result_request.user_context.user_id,
                test_execute_plan_request.user_context.user_id,
            )

        finally:
            mock_session_controller_client_instance.terminate_session.return_value = (
                mock.Mock()
            )
            self.stopSession(mock_session_controller_client_instance, session)

    def test_sql_lazy_transformation(self):
        test_uuid = "f728f1b4-00a7-4e6e-8365-d12b4a7d42ab"
        test_execute_plan_request_1: ExecutePlanRequest = ExecutePlanRequest(
            session_id="mock-session_id-from-super",
            client_type="mock-client_type-from-super",
            plan=Plan(command=Command(sql_command=SqlCommand(sql="SELECT 1"))),
            tags=["mock-tag-from-super"],
            user_context=UserContext(user_id="mock-user-from-super"),
            operation_id=test_uuid,
        )
        test_execute_plan_request_2: ExecutePlanRequest = ExecutePlanRequest(
            session_id="mock-session_id-from-super",
            client_type="mock-client_type-from-super",
            plan=Plan(
                command=Command(
                    sql_command=SqlCommand(sql="INSERT INTO test_table_2 ...")
                )
            ),
            tags=["mock-tag-from-super"],
            user_context=UserContext(user_id="mock-user-from-super"),
            operation_id=test_uuid,
        )
        test_execute_plan_request_3: ExecutePlanRequest = ExecutePlanRequest(
            session_id="mock-session_id-from-super",
            client_type="mock-client_type-from-super",
            plan=Plan(
                command=Command(
                    sql_command=SqlCommand(
                        sql="DROP TABLE IF EXISTS selections"
                    )
                )
            ),
            tags=["mock-tag-from-super"],
            user_context=UserContext(user_id="mock-user-from-super"),
            operation_id=test_uuid,
        )

        self.assertTrue(
            DataprocSparkSession._sql_lazy_transformation(
                test_execute_plan_request_1
            )
        )
        self.assertFalse(
            DataprocSparkSession._sql_lazy_transformation(
                test_execute_plan_request_2
            )
        )
        self.assertFalse(
            DataprocSparkSession._sql_lazy_transformation(
                test_execute_plan_request_3
            )
        )

    @mock.patch("google.auth.default")
    @mock.patch("google.cloud.dataproc_v1.SessionControllerClient")
    @mock.patch("pyspark.sql.connect.client.SparkConnectClient.config")
    @mock.patch(
        "google.cloud.dataproc_spark_connect.DataprocSparkSession.Builder.generate_dataproc_session_id"
    )
    @mock.patch(
        "google.cloud.dataproc_spark_connect.session.is_s8s_session_active"
    )
    def test_builder_pattern_runtime_config(
        self,
        mock_is_s8s_session_active,
        mock_dataproc_session_id,
        mock_client_config,
        mock_session_controller_client,
        mock_credentials,
    ):
        session = None
        mock_session_controller_client_instance = (
            self._setup_session_creation_mocks(
                mock_is_s8s_session_active,
                mock_dataproc_session_id,
                mock_client_config,
                mock_session_controller_client,
                mock_credentials,
            )
        )

        try:
            session = (
                DataprocSparkSession.builder.runtimeVersion("3.0")
                .config(
                    "spark.executor.cores", "8"
                )  # Use existing Spark config method
                .config("spark.executor.memory", "4g")
                .config("spark.sql.adaptive.enabled", "true")
                .getOrCreate()
            )

            # Verify the session was created with the correct runtime config
            create_session_request = mock_session_controller_client_instance.create_session.call_args[
                0
            ][
                0
            ]
            self.assertEqual(
                create_session_request.session.runtime_config.version, "3.0"
            )
            # Note: Spark configs are handled through existing Spark mechanisms
            # The key is that runtimeVersion works correctly

        finally:
            mock_session_controller_client_instance.terminate_session.return_value = (
                mock.Mock()
            )
            self.stopSession(mock_session_controller_client_instance, session)

    @mock.patch("google.auth.default")
    @mock.patch("google.cloud.dataproc_v1.SessionControllerClient")
    @mock.patch("pyspark.sql.connect.client.SparkConnectClient.config")
    @mock.patch(
        "google.cloud.dataproc_spark_connect.DataprocSparkSession.Builder.generate_dataproc_session_id"
    )
    @mock.patch(
        "google.cloud.dataproc_spark_connect.session.is_s8s_session_active"
    )
    def test_builder_pattern_environment_config(
        self,
        mock_is_s8s_session_active,
        mock_dataproc_session_id,
        mock_client_config,
        mock_session_controller_client,
        mock_credentials,
    ):
        session = None
        mock_session_controller_client_instance = (
            self._setup_session_creation_mocks(
                mock_is_s8s_session_active,
                mock_dataproc_session_id,
                mock_client_config,
                mock_session_controller_client,
                mock_credentials,
            )
        )

        try:
            session = (
                DataprocSparkSession.builder.serviceAccount(
                    "test-service@project.iam.gserviceaccount.com"
                )
                .subnetwork(
                    "projects/test-project/regions/us-central1/subnetworks/test-subnet"
                )
                .ttlSeconds(3600)
                .idleTtlSeconds(1800)
                .getOrCreate()
            )

            # Verify the session was created with the correct environment config
            create_session_request = mock_session_controller_client_instance.create_session.call_args[
                0
            ][
                0
            ]
            self.assertEqual(
                create_session_request.session.environment_config.execution_config.service_account,
                "test-service@project.iam.gserviceaccount.com",
            )
            self.assertEqual(
                create_session_request.session.environment_config.execution_config.subnetwork_uri,
                "projects/test-project/regions/us-central1/subnetworks/test-subnet",
            )
            # TTL can be represented as either timedelta or dict with seconds
            ttl_value = (
                create_session_request.session.environment_config.execution_config.ttl
            )
            if isinstance(ttl_value, datetime.timedelta):
                self.assertEqual(ttl_value.total_seconds(), 3600)
            else:
                self.assertEqual(ttl_value, {"seconds": 3600})

            idle_ttl_value = (
                create_session_request.session.environment_config.execution_config.idle_ttl
            )
            if isinstance(idle_ttl_value, datetime.timedelta):
                self.assertEqual(idle_ttl_value.total_seconds(), 1800)
            else:
                self.assertEqual(idle_ttl_value, {"seconds": 1800})

        finally:
            mock_session_controller_client_instance.terminate_session.return_value = (
                mock.Mock()
            )
            self.stopSession(mock_session_controller_client_instance, session)

    @mock.patch("google.auth.default")
    @mock.patch("google.cloud.dataproc_v1.SessionControllerClient")
    @mock.patch("pyspark.sql.connect.client.SparkConnectClient.config")
    @mock.patch(
        "google.cloud.dataproc_spark_connect.DataprocSparkSession.Builder.generate_dataproc_session_id"
    )
    @mock.patch(
        "google.cloud.dataproc_spark_connect.session.is_s8s_session_active"
    )
    def test_service_account_sets_auth_type_automatically(
        self,
        mock_is_s8s_session_active,
        mock_dataproc_session_id,
        mock_client_config,
        mock_session_controller_client,
        mock_credentials,
    ):
        """Test that setting a service account automatically sets auth type to SERVICE_ACCOUNT."""
        session = None
        mock_session_controller_client_instance = (
            self._setup_session_creation_mocks(
                mock_is_s8s_session_active,
                mock_dataproc_session_id,
                mock_client_config,
                mock_session_controller_client,
                mock_credentials,
            )
        )

        try:
            session = DataprocSparkSession.builder.serviceAccount(
                "test-service@project.iam.gserviceaccount.com"
            ).getOrCreate()

            # Verify the session was created with the correct authentication config
            create_session_request = mock_session_controller_client_instance.create_session.call_args[
                0
            ][
                0
            ]
            exec_config = (
                create_session_request.session.environment_config.execution_config
            )
            self.assertEqual(
                exec_config.service_account,
                "test-service@project.iam.gserviceaccount.com",
            )
            # Verify that authentication type is automatically set to SERVICE_ACCOUNT
            self.assertEqual(
                exec_config.authentication_config.user_workload_authentication_type,
                AuthenticationConfig.AuthenticationType.SERVICE_ACCOUNT,
            )

        finally:
            mock_session_controller_client_instance.terminate_session.return_value = (
                mock.Mock()
            )
            self.stopSession(mock_session_controller_client_instance, session)

    @mock.patch("google.auth.default")
    @mock.patch("google.cloud.dataproc_v1.SessionControllerClient")
    @mock.patch("pyspark.sql.connect.client.SparkConnectClient.config")
    @mock.patch(
        "google.cloud.dataproc_spark_connect.DataprocSparkSession.Builder.generate_dataproc_session_id"
    )
    @mock.patch(
        "google.cloud.dataproc_spark_connect.session.is_s8s_session_active"
    )
    def test_builder_pattern_ttl_with_timedelta(
        self,
        mock_is_s8s_session_active,
        mock_dataproc_session_id,
        mock_client_config,
        mock_session_controller_client,
        mock_credentials,
    ):
        session = None
        mock_session_controller_client_instance = (
            self._setup_session_creation_mocks(
                mock_is_s8s_session_active,
                mock_dataproc_session_id,
                mock_client_config,
                mock_session_controller_client,
                mock_credentials,
            )
        )

        try:
            # Test using timedelta objects
            session = (
                DataprocSparkSession.builder.ttl(datetime.timedelta(hours=1))
                .idleTtl(datetime.timedelta(minutes=30))
                .getOrCreate()
            )

            # Verify the session was created with the correct TTL values
            create_session_request = mock_session_controller_client_instance.create_session.call_args[
                0
            ][
                0
            ]

            # TTL should be 3600 seconds (1 hour)
            ttl_value = (
                create_session_request.session.environment_config.execution_config.ttl
            )
            if isinstance(ttl_value, datetime.timedelta):
                self.assertEqual(ttl_value.total_seconds(), 3600)
            else:
                self.assertEqual(ttl_value, {"seconds": 3600})

            # Idle TTL should be 1800 seconds (30 minutes)
            idle_ttl_value = (
                create_session_request.session.environment_config.execution_config.idle_ttl
            )
            if isinstance(idle_ttl_value, datetime.timedelta):
                self.assertEqual(idle_ttl_value.total_seconds(), 1800)
            else:
                self.assertEqual(idle_ttl_value, {"seconds": 1800})

        finally:
            mock_session_controller_client_instance.terminate_session.return_value = (
                mock.Mock()
            )
            self.stopSession(mock_session_controller_client_instance, session)

    @mock.patch("google.auth.default")
    @mock.patch("google.cloud.dataproc_v1.SessionControllerClient")
    @mock.patch("pyspark.sql.connect.client.SparkConnectClient.config")
    @mock.patch(
        "google.cloud.dataproc_spark_connect.DataprocSparkSession.Builder.generate_dataproc_session_id"
    )
    @mock.patch(
        "google.cloud.dataproc_spark_connect.session.is_s8s_session_active"
    )
    def test_builder_pattern_session_template_and_labels(
        self,
        mock_is_s8s_session_active,
        mock_dataproc_session_id,
        mock_client_config,
        mock_session_controller_client,
        mock_credentials,
    ):
        session = None
        mock_session_controller_client_instance = (
            self._setup_session_creation_mocks(
                mock_is_s8s_session_active,
                mock_dataproc_session_id,
                mock_client_config,
                mock_session_controller_client,
                mock_credentials,
            )
        )

        try:
            session = (
                DataprocSparkSession.builder.sessionTemplate(
                    "projects/test-project/locations/us-central1/sessionTemplates/test-template"
                )
                .label("environment", "production")
                .label("team", "data-science")
                .labels({"cost-center": "engineering", "version": "1.0"})
                .getOrCreate()
            )

            # Verify the session was created with the correct session template and labels
            create_session_request = mock_session_controller_client_instance.create_session.call_args[
                0
            ][
                0
            ]
            self.assertEqual(
                create_session_request.session.session_template,
                "projects/test-project/locations/us-central1/sessionTemplates/test-template",
            )
            self.assertEqual(
                create_session_request.session.labels["environment"],
                "production",
            )
            self.assertEqual(
                create_session_request.session.labels["team"], "data-science"
            )
            self.assertEqual(
                create_session_request.session.labels["cost-center"],
                "engineering",
            )
            self.assertEqual(
                create_session_request.session.labels["version"], "1.0"
            )

        finally:
            mock_session_controller_client_instance.terminate_session.return_value = (
                mock.Mock()
            )
            self.stopSession(mock_session_controller_client_instance, session)

    @mock.patch("google.auth.default")
    @mock.patch("google.cloud.dataproc_v1.SessionControllerClient")
    @mock.patch("pyspark.sql.connect.client.SparkConnectClient.config")
    @mock.patch(
        "google.cloud.dataproc_spark_connect.DataprocSparkSession.Builder.generate_dataproc_session_id"
    )
    @mock.patch(
        "google.cloud.dataproc_spark_connect.session.is_s8s_session_active"
    )
    def test_builder_pattern_combined_with_dataprocSessionConfig(
        self,
        mock_is_s8s_session_active,
        mock_dataproc_session_id,
        mock_client_config,
        mock_session_controller_client,
        mock_credentials,
    ):
        session = None
        mock_session_controller_client_instance = (
            self._setup_session_creation_mocks(
                mock_is_s8s_session_active,
                mock_dataproc_session_id,
                mock_client_config,
                mock_session_controller_client,
                mock_credentials,
            )
        )

        try:
            # Test combining dataprocSessionConfig with builder pattern methods
            base_config = Session()
            base_config.runtime_config.version = "3.0"
            base_config.runtime_config.properties["spark.executor.cores"] = "4"
            base_config.labels["base-label"] = "base-value"

            session = (
                DataprocSparkSession.builder.dataprocSessionConfig(base_config)
                .config(
                    "spark.executor.cores", "8"
                )  # Override using existing Spark method
                .config(
                    "spark.executor.memory", "4g"
                )  # Add new using existing Spark method
                .label("additional-label", "additional-value")  # Add new
                .getOrCreate()
            )

            # Verify the session was created with combined config
            create_session_request = mock_session_controller_client_instance.create_session.call_args[
                0
            ][
                0
            ]
            self.assertEqual(
                create_session_request.session.runtime_config.version, "3.0"
            )
            self.assertEqual(
                create_session_request.session.labels["base-label"],
                "base-value",
            )  # From base config
            self.assertEqual(
                create_session_request.session.labels["additional-label"],
                "additional-value",
            )  # Added

        finally:
            mock_session_controller_client_instance.terminate_session.return_value = (
                mock.Mock()
            )
            self.stopSession(mock_session_controller_client_instance, session)

    @mock.patch("google.auth.default")
    @mock.patch("google.cloud.dataproc_v1.SessionControllerClient")
    @mock.patch("pyspark.sql.connect.client.SparkConnectClient.config")
    @mock.patch(
        "google.cloud.dataproc_spark_connect.DataprocSparkSession.Builder.generate_dataproc_session_id"
    )
    @mock.patch(
        "google.cloud.dataproc_spark_connect.session.is_s8s_session_active"
    )
    @mock.patch("google.cloud.dataproc_spark_connect.session.logger")
    def test_builder_pattern_system_label_protection(
        self,
        mock_logger,
        mock_is_s8s_session_active,
        mock_dataproc_session_id,
        mock_client_config,
        mock_session_controller_client,
        mock_credentials,
    ):
        session = None
        mock_session_controller_client_instance = (
            self._setup_session_creation_mocks(
                mock_is_s8s_session_active,
                mock_dataproc_session_id,
                mock_client_config,
                mock_session_controller_client,
                mock_credentials,
            )
        )

        try:
            session = (
                DataprocSparkSession.builder.label(
                    "dataproc-session-client", "malicious-override"
                )  # Try to override system label
                .label(
                    "goog-colab-notebook-id", "fake-notebook"
                )  # Try to override system label
                .label("user-label", "allowed-value")  # This should work
                .labels(
                    {
                        "dataproc-session-client": "another-attempt",
                        "valid-label": "valid-value",
                    }
                )
                .getOrCreate()
            )

            # Verify system labels were protected
            create_session_request = mock_session_controller_client_instance.create_session.call_args[
                0
            ][
                0
            ]

            # System labels should not be overridden
            self.assertNotEqual(
                create_session_request.session.labels.get(
                    "dataproc-session-client"
                ),
                "malicious-override",
            )
            self.assertNotEqual(
                create_session_request.session.labels.get(
                    "goog-colab-notebook-id"
                ),
                "fake-notebook",
            )

            # User labels should be allowed
            self.assertEqual(
                create_session_request.session.labels["user-label"],
                "allowed-value",
            )
            self.assertEqual(
                create_session_request.session.labels["valid-label"],
                "valid-value",
            )

            # Verify warnings were logged
            expected_calls = [
                mock.call(
                    "Label 'dataproc-session-client' is a system label and cannot be overridden by user. Ignoring."
                ),
                mock.call(
                    "Label 'goog-colab-notebook-id' is a system label and cannot be overridden by user. Ignoring."
                ),
                mock.call(
                    "Label 'dataproc-session-client' is a system label and cannot be overridden by user. Ignoring."
                ),
            ]
            mock_logger.warning.assert_has_calls(expected_calls, any_order=True)

        finally:
            mock_session_controller_client_instance.terminate_session.return_value = (
                mock.Mock()
            )
            self.stopSession(mock_session_controller_client_instance, session)

    @mock.patch.dict(os.environ, {}, clear=True)
    @mock.patch("google.auth.default")
    @mock.patch("google.cloud.dataproc_v1.SessionControllerClient")
    @mock.patch("pyspark.sql.connect.client.SparkConnectClient.config")
    @mock.patch(
        "google.cloud.dataproc_spark_connect.DataprocSparkSession.Builder.generate_dataproc_session_id"
    )
    @mock.patch(
        "google.cloud.dataproc_spark_connect.session.is_s8s_session_active"
    )
    @mock.patch(
        "google.cloud.dataproc_spark_connect.environment.get_client_environment_label"
    )
    def test_create_session_with_client_environment_label(
        self,
        mock_get_client_environment_label,
        mock_is_s8s_session_active,
        mock_dataproc_session_id,
        mock_client_config,
        mock_session_controller_client,
        mock_credentials,
    ):
        """Tests that the client environment label is correctly added to the session request."""
        # Setup common mocks
        mock_is_s8s_session_active.return_value = True
        mock_session_controller_client_instance = (
            mock_session_controller_client.return_value
        )
        mock_dataproc_session_id.return_value = (
            "6fa459ea-ee8a-3ca4-894e-db77e160355e"
        )
        mock_client_config.return_value = ConfigResult.fromProto(
            ConfigResponse()
        )
        cred = mock.MagicMock()
        cred.token = "token"
        mock_credentials.return_value = (cred, "")

        environments_to_test = [
            "colab-enterprise",
            "colab",
            "workbench-jupyter",
            "vscode",
            "jetbrains",
            "jupyter",
            "unknown",
        ]

        for env_label in environments_to_test:
            with self.subTest(env=env_label):
                session = None
                mock_session_controller_client_instance.create_session.reset_mock()
                mock_get_client_environment_label.reset_mock()

                # Set mock returns for this specific subtest
                mock_get_client_environment_label.return_value = env_label
                mock_operation = mock.Mock()
                session_response = Session()
                session_response.runtime_info.endpoints = {
                    "Spark Connect Server": "sc://spark-connect-server.example.com:443"
                }
                session_response.uuid = "6fa459ea-ee8a-3ca4-894e-db77e160355e"
                mock_operation.result.side_effect = [session_response]
                mock_session_controller_client_instance.create_session.return_value = (
                    mock_operation
                )

                # Build the expected request for this subtest
                expected_request = CreateSessionRequest()
                expected_request.parent = (
                    "projects/test-project/locations/test-region"
                )
                expected_request.session_id = (
                    "6fa459ea-ee8a-3ca4-894e-db77e160355e"
                )
                expected_request.session.name = "projects/test-project/locations/test-region/sessions/6fa459ea-ee8a-3ca4-894e-db77e160355e"
                expected_request.session.runtime_config.version = "3.0"
                expected_request.session.spark_connect_session = (
                    SparkConnectConfig()
                )
                # This is the crucial part of the test
                expected_request.session.labels["dataproc-session-client"] = (
                    env_label
                )

                # If the environment has a subnet configured, add it to the expected request.
                subnet = os.getenv("DATAPROC_SPARK_CONNECT_SUBNET")
                if subnet:
                    expected_request.session.environment_config.execution_config.subnetwork_uri = (
                        subnet
                    )

                try:
                    # Reset singleton state before each subtest run
                    DataprocSparkSession._active_s8s_session_id = None
                    DataprocSparkSession._default_session = None

                    # Set up project and region for the builder
                    session = (
                        DataprocSparkSession.builder.projectId("test-project")
                        .location("test-region")
                        .getOrCreate()
                    )

                    mock_get_client_environment_label.assert_called_once()
                    mock_session_controller_client_instance.create_session.assert_called_once_with(
                        expected_request
                    )
                finally:
                    if session:
                        self.stopSession(
                            mock_session_controller_client_instance, session
                        )

    @mock.patch("google.auth.default")
    @mock.patch("google.cloud.dataproc_v1.SessionControllerClient")
    @mock.patch("pyspark.sql.connect.client.SparkConnectClient.config")
    @mock.patch(
        "google.cloud.dataproc_spark_connect.DataprocSparkSession.Builder.generate_dataproc_session_id"
    )
    @mock.patch(
        "google.cloud.dataproc_spark_connect.session.is_s8s_session_active"
    )
    def test_execution_progress_handler(
        self,
        mock_is_s8s_session_active,
        mock_dataproc_session_id,
        mock_client_config,
        mock_session_controller_client,
        mock_credentials,
    ):
        session = None
        mock_is_s8s_session_active.return_value = True
        mock_session_controller_client_instance = (
            mock_session_controller_client.return_value
        )
        mock_dataproc_session_id.return_value = "sc-20240702-103952-abcdef"
        mock_client_config.return_value = ConfigResult.fromProto(
            ConfigResponse()
        )
        cred = mock.MagicMock()
        cred.token = "token"
        mock_credentials.return_value = (cred, "")
        mock_operation = mock.Mock()
        session_response = Session()
        session_response.runtime_info.endpoints = {
            "Spark Connect Server": "sc://spark-connect-server.example.com:443"
        }
        session_response.uuid = "c002e4ef-fe5e-41a8-a157-160aa73e4f7f"
        mock_operation.result.side_effect = [session_response]
        mock_session_controller_client_instance.create_session.return_value = (
            mock_operation
        )

        try:
            session = DataprocSparkSession.builder.getOrCreate()
            client = session.client

            # By default Dataproc handler is registered
            self.assertEqual(len(client._progress_handlers), 1)

            # Dataproc handler isn't cleared with clearProgressHandlers() method
            session.clearProgressHandlers()
            self.assertEqual(len(client._progress_handlers), 1)

        finally:
            mock_session_controller_client_instance.terminate_session.return_value = (
                mock.Mock()
            )
            self.stopSession(mock_session_controller_client_instance, session)

    @mock.patch("time.sleep", return_value=None)
    @mock.patch("google.cloud.dataproc_v1.SessionControllerClient")
    def test_wait_for_session_available_success(
        self, mock_session_controller_client, mock_sleep
    ):
        """Test that the method waits and returns the session when the endpoint appears."""
        mock_client = mock_session_controller_client.return_value
        session_name = (
            "projects/test-project/locations/test-region/sessions/test-session"
        )

        # Session without the endpoint
        session_pending = Session()
        session_pending.name = session_name

        # Session with the endpoint
        session_ready = Session()
        session_ready.name = session_name
        session_ready.runtime_info.endpoints["Spark Connect Server"] = (
            "sc://example.com:443"
        )

        # Mock get_session to return pending, then ready
        mock_client.get_session.side_effect = [
            session_pending,
            session_pending,
            session_ready,
        ]

        builder = DataprocSparkSession.Builder()
        builder._session_controller_client = (
            mock_client  # Inject the mock client
        )

        result = builder._wait_for_session_available(session_name, timeout=10)

        self.assertEqual(result, session_ready)
        self.assertEqual(mock_client.get_session.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)

    @mock.patch("time.sleep", return_value=None)
    @mock.patch("google.cloud.dataproc_v1.SessionControllerClient")
    def test_wait_for_session_available_timeout(
        self, mock_session_controller_client, mock_sleep
    ):
        """Test that the method raises RuntimeError on timeout."""
        mock_client = mock_session_controller_client.return_value
        session_name = (
            "projects/test-project/locations/test-region/sessions/test-session"
        )

        # Session that never gets the endpoint
        session_pending = Session()
        session_pending.name = session_name

        mock_client.get_session.return_value = session_pending

        builder = DataprocSparkSession.Builder()
        builder._session_controller_client = (
            mock_client  # Inject the mock client
        )

        with self.assertRaises(RuntimeError) as context:
            # Use a short timeout for the test
            builder._wait_for_session_available(session_name, timeout=1)

        self.assertIn(
            f"Spark Connect endpoint not available for session {session_name}",
            str(context.exception),
        )


class SessionIdValidationTests(unittest.TestCase):
    """Test cases for session ID validation and custom session ID functionality."""

    def test_valid_session_ids(self):
        """Test that valid session IDs pass validation."""
        valid_ids = [
            "test-session",
            "mysession123",
            "a-b-c-d",
            "session-2024-01-01",
            "spark-session-1",
            "a" * 63,  # Max length
            "abcd",  # Min length (4 chars)
        ]
        for session_id in valid_ids:
            self.assertTrue(
                _is_valid_session_id(session_id),
                f"Session ID '{session_id}' should be valid",
            )

    def test_invalid_session_ids(self):
        """Test that invalid session IDs fail validation."""
        invalid_ids = [
            "",  # Empty
            "123-session",  # Starts with number
            "Session",  # Contains uppercase
            "session_name",  # Contains underscore
            "session-",  # Ends with hyphen
            "-session",  # Starts with hyphen
            "abc",  # Too short (< 4 chars)
            "a" * 64,  # Too long (> 63 chars)
            "session name",  # Contains space
            "session.name",  # Contains period
        ]
        for session_id in invalid_ids:
            self.assertFalse(
                _is_valid_session_id(session_id),
                f"Session ID '{session_id}' should be invalid",
            )

    def test_dataproc_session_id_builder_method(self):
        """Test the dataprocSessionId() builder method."""
        builder = DataprocSparkSession.builder

        # Test valid session ID
        result = builder.dataprocSessionId("test-session")
        self.assertEqual(builder._custom_session_id, "test-session")
        self.assertEqual(result, builder)  # Check method chaining

        # Test invalid session ID raises ValueError
        with self.assertRaises(ValueError) as context:
            builder.dataprocSessionId("123-invalid")
        self.assertIn("Invalid session ID", str(context.exception))

    @mock.patch(
        "google.cloud.dataproc_spark_connect.session.SessionControllerClient"
    )
    def test_session_reuse_with_custom_id(self, mock_session_controller_client):
        """Test that sessions are reused when custom ID is provided."""
        mock_client = mock_session_controller_client.return_value

        # Setup mock session in ACTIVE state
        active_session = Session()
        active_session.state = Session.State.ACTIVE
        active_session.uuid = "test-uuid"
        active_session.runtime_info.endpoints = {
            "Spark Connect Server": "sc://example.com:443"
        }
        mock_client.get_session.return_value = active_session

        builder = DataprocSparkSession.Builder()
        builder._project_id = "test-project"
        builder._region = "test-region"
        builder._custom_session_id = "my-session"

        # Test that _get_session_by_id returns the active session
        result = builder._get_session_by_id("my-session")
        self.assertEqual(result, active_session)
        mock_client.get_session.assert_called_once()

    @mock.patch(
        "google.cloud.dataproc_spark_connect.session.SessionControllerClient"
    )
    def test_session_skip_terminated(self, mock_session_controller_client):
        """Test that terminated sessions are skipped, not cleaned up."""
        mock_client = mock_session_controller_client.return_value

        # Setup mock session in TERMINATED state
        terminated_session = Session()
        terminated_session.state = Session.State.TERMINATED
        mock_client.get_session.return_value = terminated_session

        builder = DataprocSparkSession.Builder()
        builder._project_id = "test-project"
        builder._region = "test-region"
        builder._custom_session_id = "my-session"

        # Test that _get_session_by_id returns None for terminated session
        result = builder._get_session_by_id("my-session")
        self.assertIsNone(result)
        mock_client.get_session.assert_called_once()


if __name__ == "__main__":
    unittest.main()
