import inspect
import pprint


class ParametrizedObject(object):
    """
    Get the object configuration from the __init__ method. 
    The same as is done in the sklearn package.
    """
    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the recommender"""
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        args, varargs, kw, default, kwonlyargs, kwonlydefaults, annotations = inspect.getfullargspec(init)
        if varargs is not None:
            raise RuntimeError(
                "No varargs allowed in __init__ of model class!"
                "Please correct: %{cls}".format(cls=cls)
            )
        args.pop(0)  # remove self
        args.sort()
        return args

    def get_params(self):
        """Get parameters for this model."""
        return {
            k: v.get_config() if isinstance(v, ParametrizedObject) else v
            for k, v in [(key, getattr(self, key, None)) for key in self._get_param_names()]
        }

    def get_config(self):
        """
        Returns dictionary representation for model configuration
        :return: dict
        """
        conf = dict(name=self.__class__.__name__)
        conf.update(self.get_params())
        return conf

    def __repr__(self):
        class_name = self.__class__.__name__
        return '%s(%s)' % (class_name, pprint.pformat(self.get_params())[1:-1])
