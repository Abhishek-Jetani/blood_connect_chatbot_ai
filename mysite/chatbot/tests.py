import unittest
from .views import faq_answer


class FAQModelTest(unittest.TestCase):
    def test_under_18_question(self):
        q = "Can I donate blood if I'm 17 years old?"
        ans = faq_answer(q)
        self.assertIsNotNone(ans)
        self.assertIn('18', ans)  # expects age mention

    def test_tattoo_question(self):
        q = "I got a tattoo last month, can I donate?"
        ans = faq_answer(q)
        self.assertIsNotNone(ans)
        self.assertIn('tattoo', ans.lower())


if __name__ == '__main__':
    unittest.main()
